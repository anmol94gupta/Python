#two number sum
def twoNumberSum(array, targetSum):
    # Order n^2
	q=0
	p=1
	o=[]
	for n in range (q,len(array)):
		for m in range (p,len(array)):
			if array[n]+array[m]==targetSum:
				o.append(array[n])
				o.append(array[m])
		p=p+1
		q=q+1
	return(o)