from neo4j import GraphDatabase
from api.config.database import Config
import logging

class Neo4jService:
    def __init__(self):
        self._driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
    
    def close(self):
        self._driver.close()
    
    def get_medical_knowledge(self, query):
        """
        从知识图谱中检索医学知识
        Retrieve medical knowledge from the knowledge graph
        """
        with self._driver.session() as session:
            result = session.read_transaction(self._search_knowledge, query)
            return result
    
    @staticmethod
    def _search_knowledge(tx, query):
        # 这里是一个示例Cypher查询，根据实际图结构进行调整
        cypher_query = """
        MATCH (n)
        WHERE n.name CONTAINS $query OR n.description CONTAINS $query
        RETURN n LIMIT 10
        """
        result = tx.run(cypher_query, parameters={"query": query})
        return [record["n"] for record in result]
    
    def save_feedback(self, question, answer, feedback_score):
        """
        保存用户反馈数据用于模型微调
        Save user feedback data for model fine-tuning
        """
        with self._driver.session() as session:
            session.write_transaction(self._store_feedback, question, answer, feedback_score)
    
    @staticmethod
    def _store_feedback(tx, question, answer, feedback_score):
        cypher_query = """
        CREATE (f:Feedback {
            question: $question,
            answer: $answer,
            score: $feedback_score,
            timestamp: timestamp()
        })
        RETURN f
        """
        result = tx.run(cypher_query, 
                       parameters={"question": question, 
                                "answer": answer, 
                                "feedback_score": feedback_score})
        return result.single()

# Global instance
neo4j_service = Neo4jService()