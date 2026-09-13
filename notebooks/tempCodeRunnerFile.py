print("\nRECOMMENDED WORK EXAMPLES:")
print(recommended["work"].head(10).to_list())

print("\nSANCTIONED WORK EXAMPLES:")
print(sanctioned["work"].head(10).to_list())

print("\nCOMPLETED WORK EXAMPLES:")
print(completed["work"].head(10).to_list())

print("\nEXPENDITURE WORK EXAMPLES:")
print(expenditure["work"].head(10).to_list())

print("\nEXPENDITURE WORK ID EXAMPLES:")
print(expenditure["work_id"].head(10).to_list())