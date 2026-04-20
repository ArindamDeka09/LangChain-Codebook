from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Adam"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt = 0,  lt = 4, default = 5) 
    

new_student = {'age': 23, 'email': 'adam@gmail.com', 'cgpa': 3.5, 'description': "A decimal value representing the student's CGPA"}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()
print(student_json)