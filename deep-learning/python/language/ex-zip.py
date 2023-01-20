print("simple case:")
languages = ['Java', 'Python', 'JavaScript']
versions = [14, 3, 6]
result = zip(languages, versions)
print(list(result))

print("advanced case:")
arr_w = [[-2], [-3], [4]]
arr_b = [-1, -2, -3]
y_label = ['yhat=0', 'yhat=1', 'yhat=2']
y_color = ['b', 'r', 'g']
r=zip(arr_w,arr_b, y_label, y_color)
for w, b, y_l, y_c in r:
    print(w,b, y_l, y_c)
