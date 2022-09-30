

l = [1,5,3,0,9,2,3]

def find_min(l, new_l):
	n = len(l)

	minm=100000000
	for i in range(n):
		if l[i] in new_l: continue
		if l[i] < minm:
			minm = l[i]

	return minm

# print(find_min(l))


new_l = []
for i in range(len(l)):
	minm = find_min(l, new_l)
	new_l.append(minm)

print(new_l[:-1])




# A = l

# for i in range(len(l)):
# 	for j in range(len(l)):
# 		if (A[i] < A[j]):
# 			temp = A[i]
# 			A[i] = A[j]
# 			A[j] = temp

# print(A[len(A)//2])