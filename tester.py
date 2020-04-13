import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

print(ncr(5,1))
file = open('test_file', 'r')
out = open('out', 'w')

lines = file.readlines()
lines = [x.strip() for x in lines]
ind = 0

while ind < len(lines):
    exp = int(lines[ind])
    ind += 1
    #print(exp)

    nums = []
    while lines[ind] != '***':

        nums.append(list(map(int, lines[ind].split(' '))))
        ind += 1
    
    tot = 0
    #print(nums)
    for pair in nums:
        tot += ncr(pair[0] - 1, pair[1] - 1)

    if tot != exp:
        out.write(f"Failed for {exp}, got {tot}\n")
    else:
        out.write(f"Passed for {exp}\n")
    
    ind += 1

