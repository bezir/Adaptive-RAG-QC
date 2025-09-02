local config = import "qa_base.jsonnet";

config {
    "start_state": "step_by_step_bm25_retriever",
    "end_state": "[EOQ]",
    "models": {
        "step_by_step_bm25_retriever": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "step_by_step_cot_reasoning_gen",
            "retrieval_type": "bm25",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": 5,
            "global_max_num_paras": 15,
            "query_source": "question_or_last_generated_sentence",
            "source_corpus_name": "triviaqa",
            "document_type": "title_paragraph_text",
            "return_pids": false,
            "cumulate_titles": true,
            "end_state": "[EOQ]",
        },
        "step_by_step_cot_reasoning_gen": {
            "name": "step_by_step_cot_gen",
            "next_model": "step_by_step_exit_controller",
            "prompt_file": "prompts/trivia/qa_cot_gemini.txt",
            "question_prefix": "Answer the following question by reasoning step-by-step.\n",
            "prompt_reader_args": {},
            "generation_type": "sentences",
            "reset_queries_as_sentences": false,
            "add_context": true,
            "shuffle_paras": false,
            "terminal_return_type": null,
            "disable_exit": true,
            "end_state": "[EOQ]",
            "gen_model": "gemini",
            "model_name": "gemini-2.5-flash-lite",
            "model_tokens_limit": 128000,
            "max_length": 200,
        },
        "step_by_step_exit_controller": {
            "name": "step_by_step_exit_controller",
            "end_state": "[EOQ]",
        },
    }
}
