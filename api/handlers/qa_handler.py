from flask import Blueprint, request, jsonify
from api.models.medical_model import medical_model
from api.services.neo4j_service import neo4j_service
from api.utils.model_utils import ConversationContext, calculate_model_score
import logging

qa_bp = Blueprint('qa', __name__)

# 为每个会话维护对话上下文
conversation_contexts = {}

@qa_bp.route('/qa/ask', methods=['POST'])
def ask_medical_question():
    """
    处理医疗问答请求
    Handle medical QA request
    """
    try:
        data = request.get_json()
        question = data.get('question')
        session_id = data.get('session_id', 'default')
        expected_keywords = data.get('expected_keywords', [])
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # 获取或创建对话上下文
        if session_id not in conversation_contexts:
            conversation_contexts[session_id] = ConversationContext()
        
        context_manager = conversation_contexts[session_id]
        context = context_manager.get_context()
        
        # 从知识图谱获取相关知识
        knowledge = neo4j_service.get_medical_knowledge(question)
        knowledge_context = " ".join([str(k) for k in knowledge]) if knowledge else ""
        
        # 合并对话历史和知识图谱上下文
        full_context = f"{context} {knowledge_context}".strip()
        
        # 生成答案
        answer = medical_model.generate_answer(question, full_context)
        
        # 计算模型评分
        score = calculate_model_score(answer, expected_keywords)
        
        # 保存对话历史
        context_manager.add_turn(question, answer)
        
        return jsonify({
            'question': question,
            'answer': answer,
            'score': score,
            'session_id': session_id
        })
        
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@qa_bp.route('/qa/feedback', methods=['POST'])
def submit_feedback():
    """
    提交反馈用于模型微调
    Submit feedback for model fine-tuning
    """
    try:
        data = request.get_json()
        question = data.get('question')
        answer = data.get('answer')
        feedback_score = data.get('score')
        session_id = data.get('session_id', 'default')
        
        if not all([question, answer, feedback_score is not None]):
            return jsonify({'error': 'Question, answer, and score are required'}), 400
        
        # 保存反馈到知识图谱
        neo4j_service.save_feedback(question, answer, feedback_score)
        
        return jsonify({'message': 'Feedback saved successfully'})
        
    except Exception as e:
        logging.error(f"Error saving feedback: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@qa_bp.route('/qa/context/<session_id>', methods=['DELETE'])
def clear_context(session_id):
    """
    清除指定会话的对话上下文
    Clear conversation context for a session
    """
    if session_id in conversation_contexts:
        conversation_contexts[session_id].clear()
        del conversation_contexts[session_id]
    
    return jsonify({'message': 'Context cleared successfully'})

@qa_bp.route('/qa/health', methods=['GET'])
def health_check():
    """
    健康检查端点
    Health check endpoint
    """
    return jsonify({'status': 'ok', 'message': 'QA service is running'})