from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

def define_models(r_db):
    class Source(r_db.Model):
        id = Column(Integer, primary_key=True)
        vector_id = Column(Integer, nullable=False)

    class Question(r_db.Model):
        id = Column(Integer, primary_key=True)
        question = Column(String, nullable=False)
        count = Column(Integer, default=1)

    class Answer(r_db.Model):
        id = Column(Integer, primary_key=True)
        answer = Column(String, nullable=False)

    class Response(r_db.Model):
        id = Column(Integer, primary_key=True)
        question_id = Column(Integer, ForeignKey("question.id"), nullable=False)
        answer_id = Column(Integer, ForeignKey("answer.id"), nullable=False)
        likes = Column(Integer, default=0)
        dislikes = Column(Integer, default=0)
        timestamp = Column(DateTime, default=datetime.utcnow)

        question = relationship("Question")
        answer = relationship("Answer")
        sources = relationship('ResponseSource', back_populates='response')

    class ResponseSource(r_db.Model):
        id = Column(Integer, primary_key=True)
        response_id = Column(Integer, ForeignKey("response.id"), nullable=False)
        source_id = Column(Integer, ForeignKey("source.id"), nullable=False)

        response = relationship('Response', back_populates='sources')
        source = relationship('Source')

    # Return the models if needed
    return Source, Question, Answer, Response, ResponseSource

