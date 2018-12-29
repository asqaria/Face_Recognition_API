from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()


class Users(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(250), nullable=False)
    email = Column(String(250), nullable=False)
    phone = Column(String(250), nullable=False)
    image = Column(String(250), nullable=False)

    @property
    def serialize(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'image': self.image
        }


class Visitors(Base):
    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime, default=func.now())
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship(Users)

    @property
    def serialize(self):
        return {
            'id': self.id,
            'time': self.time,
            'user_id': self.user_id,
            'user': self.user
        }


class Suspicious(Base):
    __tablename__ = "suspicious"

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime, default=func.now())
    image = Column(String(250), nullable=False)

    @property
    def serialize(self):
        return {
            'id': self.id,
            'time': self.time,
            'image': self.image
        }


engine = create_engine('sqlite:///database.db')
Base.metadata.create_all(engine)
