"""
RAG engine with search, reranking, and answer generation capabilities
"""
import asyncio
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from src.config import RetrievalConfig, RRFConfig, GenerationConfig, PerformanceConfig, ChunkingConfig
from src.helpers.utils import QuickTimer, RequestLogger, ResourceManager, FastGroqClient
from src.document_processing import smart_chunk_configurable

def enhanced_reciprocal_rank_fusion(ranked_lists: List[List[int]], rrf_config: RRFConfig) -> List[tuple[int, float]]:
    """Enhanced RRF with configurable weighting"""
    weights = [rrf_config.semantic_weight, rrf_config.bm25_weight]
    k = rrf_config.k_parameter
    
    fused_scores = collections.defaultdict(float)
    
    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, doc_index in enumerate(ranked_list):
            fused_scores[doc_index] += weight * (1.0 / (k + rank + 1))
    
    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

class RAGEngine:
    """Main RAG engine class"""
    
    def __init__(self, embedding_model, reranker, groq_client: FastGroqClient, resource_manager: ResourceManager):
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.groq_client = groq_client
        self.resource_manager = resource_manager
        # default number of augmented queries
        self.num_augmented_queries = 5
    
    def hybrid_search_configurable(self, query: str, chunks: List[str], faiss_index, bm25, 
                                  retrieval_config: RetrievalConfig, rrf_config: RRFConfig) -> List[int]:
        """Configurable hybrid search"""
        # Semantic search
        with self.resource_manager.embedding_context():
            q_embedding = self.embedding_model.encode(
                query, 
                convert_to_tensor=True,
                normalize_embeddings=True
            ).cpu().numpy().reshape(1, -1)
            
            _, faiss_indices = faiss_index.search(q_embedding, k=retrieval_config.semantic_search_top_k)
            semantic_results = faiss_indices[0].tolist()

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_results = sorted(
            range(len(bm25_scores)), 
            key=lambda i: bm25_scores[i], 
            reverse=True
        )[:retrieval_config.bm25_search_top_k]

        # Configurable RRF
        fused_results = enhanced_reciprocal_rank_fusion(
            [semantic_results, bm25_results],
            rrf_config
        )
        
        return [idx for idx, score in fused_results[:retrieval_config.hybrid_fusion_top_k]]

    def parallel_search_configurable(self, questions: List[str], chunks: List[str], 
                                    faiss_index, bm25, timer: QuickTimer, req_logger: RequestLogger,
                                    retrieval_config: RetrievalConfig, rrf_config: RRFConfig,
                                    performance_config: PerformanceConfig) -> List[List[int]]:
        """Configurable parallel search"""
        with timer.time_step("Parallel Search"):
            
            if len(questions) == 1:
                results = [self.hybrid_search_configurable(
                    questions[0], chunks, faiss_index, bm25, 
                    retrieval_config, rrf_config
                )]
                req_logger.log_search_results(0, questions[0], [chunks[idx] for idx in results[0]])
                return results
            
            print(f"🔍 Parallel search for {len(questions)} questions...")
            all_results = [None] * len(questions)
            
            max_workers = min(performance_config.max_search_workers, len(questions))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(
                        self.hybrid_search_configurable, 
                        q, chunks, faiss_index, bm25, 
                        retrieval_config, rrf_config
                    ): i
                    for i, q in enumerate(questions)
                }
                
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        all_results[idx] = result
                        req_logger.log_search_results(idx, questions[idx], [chunks[i] for i in result])
                    except Exception as e:
                        print(f"❌ Search failed for question {idx + 1}: {e}")
                        req_logger.log_error("search_failed", str(e))
                        all_results[idx] = []
            
            return all_results

    async def generate_related_queries(self, question: str, generation_config: GenerationConfig) -> List[str]:
        """Use Groq to generate N related queries for augmentation."""
        prompt = (
            "You are a query expansion agent. Given a user question, generate "
            f"{self.num_augmented_queries} concise, diverse but semantically related search queries. "
            "Return them as a numbered list, one per line, without explanations.\n\n"
            f"QUESTION: {question}\n"
        )
        with self.resource_manager.groq_context():
            raw = await self.groq_client.generate_answer(prompt, generation_config)
        # Parse numbered lines
        candidates = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # remove leading numbering like "1. ", "- ", etc.
            if line[:2].isdigit() and line[1:2] == '.':
                line = line[2:].strip()
            elif line[:3].isdigit() and line[2:3] == '.':
                line = line[3:].strip()
            elif line.startswith('- '):
                line = line[2:].strip()
            candidates.append(line)
        # Deduplicate and trim to N
        dedup = []
        seen = set()
        for c in candidates:
            if c and c.lower() not in seen:
                dedup.append(c)
                seen.add(c.lower())
            if len(dedup) >= self.num_augmented_queries:
                break
        return dedup

    async def sanitize_query(self, original_question: str, augmented_queries: List[str], 
                           generation_config: GenerationConfig) -> str:
        """Send original + augmented queries to LLM to get a single refined question for answer generation."""
        prompt = f"""You are a query refinement agent. Given an original question and several related queries, create a single, refined question that best captures the user's intent and will be used to generate a comprehensive answer.

ORIGINAL QUESTION: {original_question}

RELATED QUERIES:
{chr(10).join(f"- {q}" for q in augmented_queries)}

TASK: Create ONE refined question that:
1. Captures the core intent of the original question
2. Incorporates relevant aspects from the related queries
3. Is clear, specific, and well-formed
4. Will help generate the most comprehensive answer

REFINED QUESTION:"""
        
        try:
            with self.resource_manager.groq_context():
                refined_question = await self.groq_client.generate_answer(prompt, generation_config)
                # Clean up the response
                refined_question = refined_question.strip()
                # Remove any prefix like "REFINED QUESTION:" if present
                if "REFINED QUESTION:" in refined_question:
                    refined_question = refined_question.split("REFINED QUESTION:")[-1].strip()
                return refined_question
        except Exception as e:
            print(f"⚠️ Query sanitization failed: {e}")
            # Fallback to original question
            return original_question

    async def augmented_parallel_search(self, questions: List[str], chunks: List[str],
                                        faiss_index, bm25, timer: QuickTimer, req_logger: RequestLogger,
                                        retrieval_config: RetrievalConfig, rrf_config: RRFConfig,
                                        performance_config: PerformanceConfig,
                                        generation_config: GenerationConfig) -> tuple[List[List[int]], List[str]]:
        """Augment each question with related queries and fuse results with RRF. Returns (search_results, sanitized_questions)."""
        with timer.time_step("Augmented Parallel Search"):
            results: List[List[int]] = []
            sanitized_questions: List[str] = []
            
            for qi, q in enumerate(questions):
                # base search
                base = self.hybrid_search_configurable(q, chunks, faiss_index, bm25, retrieval_config, rrf_config)
                # generate related queries
                try:
                    aug_queries = await self.generate_related_queries(q, generation_config)
                except Exception as e:
                    aug_queries = []
                    req_logger.log_error("query_augmentation", str(e))
                # log augmented queries
                try:
                    req_logger.log_augmented_queries(qi, q, aug_queries)
                except Exception:
                    pass
                
                # Query sanitization - get refined question for answer generation
                try:
                    sanitized_q = await self.sanitize_query(q, aug_queries, generation_config)
                    sanitized_questions.append(sanitized_q)
                    req_logger.logs.append({
                        "timestamp": req_logger.get_timestamp(),
                        "type": "query_sanitization",
                        "request_id": req_logger.req_id,
                        "question_index": qi,
                        "original_question": q,
                        "sanitized_question": sanitized_q,
                        "augmented_queries": aug_queries
                    })
                except Exception as e:
                    print(f"⚠️ Query sanitization failed for question {qi + 1}: {e}")
                    sanitized_questions.append(q)  # Fallback to original
                
                # run hybrid search for each augmented query
                ranked_lists: List[List[int]] = [base]
                for aq in aug_queries:
                    try:
                        aq_res = self.hybrid_search_configurable(aq, chunks, faiss_index, bm25, retrieval_config, rrf_config)
                        ranked_lists.append(aq_res)
                    except Exception as e:
                        req_logger.log_error("augmented_search_failed", str(e))
                # fuse all ranked lists with standard (unweighted) RRF
                # reuse enhanced_reciprocal_rank_fusion by treating weights equal
                # Build a temporary RRF config with equal weights by duplicating lists
                fused = collections.defaultdict(float)
                k = rrf_config.k_parameter
                for ranked in ranked_lists:
                    for rank, doc_index in enumerate(ranked):
                        fused[doc_index] += 1.0 / (k + rank + 1)
                fused_sorted = sorted(fused.items(), key=lambda item: item[1], reverse=True)
                final = [idx for idx, _ in fused_sorted[:retrieval_config.hybrid_fusion_top_k]]
                results.append(final)
                # log
                req_logger.log_search_results(qi, q, [chunks[idx] for idx in final])
            return results, sanitized_questions

    def configurable_rerank(self, questions: List[str], chunks: List[str], 
                           search_results: List[List[int]], timer: QuickTimer, req_logger: RequestLogger,
                           retrieval_config: RetrievalConfig, performance_config: PerformanceConfig) -> List[List[int]]:
        """Configurable reranking with performance options"""
        with timer.time_step("Reranking"):
            # Prepare pairs efficiently
            all_pairs = []
            pair_mapping = []
            
            for q_idx, (question, chunk_indices) in enumerate(zip(questions, search_results)):
                for chunk_idx in chunk_indices:
                    all_pairs.append([question, chunks[chunk_idx]])
                    pair_mapping.append((q_idx, chunk_idx))
            
            # Batch rerank with configurable batch size
            if all_pairs:
                with self.resource_manager.reranker_context():
                    scores = self.reranker.predict(all_pairs, batch_size=performance_config.reranking_batch_size)
            else:
                scores = []
            
            # Group and sort
            question_results = [[] for _ in questions]
            for i, (q_idx, chunk_idx) in enumerate(pair_mapping):
                score = scores[i] if i < len(scores) else 0.0
                question_results[q_idx].append((chunk_idx, score))
            
            # Return configurable top-k for each question
            final_results = []
            for results in question_results:
                results.sort(key=lambda x: x[1], reverse=True)
                final_results.append([idx for idx, _ in results[:retrieval_config.final_context_top_k]])
            
            return final_results

    async def configurable_generate_answers(self, questions: List[str], contexts: List[str], 
                                           timer: QuickTimer, req_logger: RequestLogger,
                                           generation_config: GenerationConfig, 
                                           performance_config: PerformanceConfig) -> List[str]:
        """Configurable answer generation with performance tuning"""
        with timer.time_step("Answer Generation"):
            print(f"🎯 Generating {len(questions)} answers...")
            
            async def generate_single(question: str, context: str, idx: int) -> str:
                prompt = f"""Single sentence with breifing. Based on the context, ""provide a direct one-sentence answer to the question. Find and give the exact number or fact for a question. If you find any other very relevant information, include it in your answer. Don't do thinking or reasoning, just give the answer directly.
Summarize the answer in a single sentence and keep it brief. Dont cite the context, just give the answer.If it state any act or criteria , find and give the acts or criteria in the answer.Answer from context even if it is false.
CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
                
                try:
                    with self.resource_manager.groq_context():
                        answer = await self.groq_client.generate_answer(prompt, generation_config)
                        req_logger.log_answer_generated(idx, question, answer, len(context))
                        return answer
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    req_logger.log_error("answer_generation", str(e))
                    return error_msg
            
            # Configurable concurrency control
            semaphore = asyncio.Semaphore(performance_config.max_generation_workers)
            
            async def generate_with_limit(q, c, i):
                async with semaphore:
                    return await generate_single(q, c, i)
            
            tasks = [generate_with_limit(q, c, i) for i, (q, c) in enumerate(zip(questions, contexts))]
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    error_msg = f"Error generating answer: {str(answer)}"
                    req_logger.log_error("answer_generation_exception", str(answer))
                    final_answers.append(error_msg)
                else:
                    final_answers.append(answer)
            
            return final_answers

    def build_indexes(self, chunks: List[str], timer: QuickTimer, performance_config: PerformanceConfig):
        """Build FAISS and BM25 indexes"""
        import faiss
        from rank_bm25 import BM25Okapi
        with timer.time_step("Index Building"):
            # Semantic index with configurable batch size
            with self.resource_manager.embedding_context():
                embeddings = self.embedding_model.encode(
                    chunks, 
                    convert_to_tensor=True,
                    batch_size=performance_config.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            
            faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
            faiss_index.add(embeddings.cpu().numpy())
            
            # BM25 index
            tokenized_corpus = [doc.lower().split() for doc in chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            
            return faiss_index, bm25

    def smart_chunk_documents(self, content_blocks: List[str], chunking_config: ChunkingConfig, timer: QuickTimer) -> List[str]:
        """Smart chunking of document content"""
        with timer.time_step("Smart Chunking"):
            all_chunks = []
            
            for block in content_blocks:
                if len(block) > chunking_config.chunk_size:
                    # Split large blocks
                    block_chunks = smart_chunk_configurable(block, chunking_config)
                    all_chunks.extend(block_chunks)
                else:
                    # Keep small blocks as-is
                    all_chunks.append(block)
            
            print(f"📝 Created {len(all_chunks)} chunks from {len(content_blocks)} blocks")
            return all_chunks

    def build_contexts(self, questions: List[str], chunks: List[str], reranked_results: List[List[int]], 
                      timer: QuickTimer, req_logger: RequestLogger) -> List[str]:
        """Build contexts from reranked results"""
        with timer.time_step("Context Building"):
            contexts = []
            for i, chunk_indices in enumerate(reranked_results):
                context_chunks = [chunks[idx] for idx in chunk_indices]
                context = "\n---\n".join(context_chunks)
                contexts.append(context)
                
                # Log top contexts for each question
                req_logger.log_top_contexts(i, questions[i], context_chunks)
            
            return contexts
