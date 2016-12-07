
import sys
import math
DEBUG = 0

def pearsonCoeff(dataT,labels):
    pCCs=[]
    rows = len(dataT)
    cols = len(dataT[0]);
    Ymean = 0
    Ystddev = 0
    for j in range(rows):
        Ymean += labels[j]
    Ymean /= rows
    for j in range(rows):
        Ystddev += (Ymean - labels[j])*(Ymean - labels[j])
    Ystddev = math.sqrt(Ystddev)
    for j in range(cols):
        if (j%1000==0): print('      ...Pearson Coeff: Features processed ',j)
        num = 0
        Xmean = 0
        Xstddev = 0
        for i in range(rows):
            Xmean += dataT[i][j]
        Xmean /= rows;
        for i in range(rows):
            Xstddev += (Xmean - dataT[i][j])*(Xmean - dataT[i][j])
        Xstddev = math.sqrt(Xstddev);
        for i in range(rows):
            num += (Xmean - dataT[i][j])*(Ymean - labels[i])
        den = Ystddev * Xstddev
        if (den == 0): r = 0
        else: r = num/den
        if(r < 0): r *= -1;
        pCCs.append(r);
    return pCCs

### Read Data ###

datafile = sys.argv[1]
f= open(datafile)
data = []

l=f.readline()
while(l != ''):
        a = l.split()
        l2=[]
        for j in range(0, len(a), 1):
                l2.append(int(a[j]))
        data.append(l2)
        l=f.readline()

rows = len(data)
cols = len(data[0])

#print("Rows:",rows)
#print("Cols:",cols)

f.close()

if(DEBUG):
        for i in range(0, len(data), 1):
                print(data[i])

### Read labels ###

labelfile = sys.argv[2]
f = open(labelfile)
y =[]
n = []
n.append(0)
n.append(0)

l=f.readline()
while(l != ''):
	a = l.split()
	y.append(int(a[0]))
	n[int(a[0])] += 1
	l = f.readline()
f.close()

#print("Labels:",len(y))
if(DEBUG):
	print("y:\t",y)


### Storing correlation values ###

col_correlation = []
col_correlation = pearsonCoeff(data,y)
#print("Correlation :\n",col_correlation)

### Finding correlation mean ###

corr_len = len(col_correlation)
mean_corr = 0

for i in range(0, corr_len, 1):
	mean_corr += col_correlation

mean_corr = mean_corr/corr_len
#print("Mean Corr:\t",mean_corr)

### List of selected features ###

selected = []

for i in range(0, corr_len, 1):
	if(col_correlation[i] > mean_corr):
		selected.append(i)				## Storing column index in selected ##

print("\n No. of features:\t ",len(selected))
print("\n Selected features:\n",selected)

rows = len(new_data)
cols = len(new_data[0])


### New dataset ###

new_data = []

for i in range(0, len(data), 1):
	for j in range(0, len(selected), 1):
		k = selected[j]
		new_data[i][j] = int(data[i][k])
#print("\n\nNew Data:\n",new_data)

rows = len(new_data)
cols = len(new_data[0])

### Test Data with selected features ###

testfile = sys.argv[3]
f= open(testfile)
test_data = []

l=f.readline()
while(l != ''):
        a = l.split()
        l2=[]
        for j in range(0, len(a), 1):
                l2.append(int(a[j]))
        test_data.append(l2)
        l=f.readline()

for i in range(0, len(test_data), 1):
        for j in range(0, len(selected), 1):
                k = selected[j]
                new_test[i][j] = int(test_data[i][k])
#print("\n\nNew Test Data:\n",new_test)

### Calculate means with new training data ###

m0= []
m1= []

for i in range (0, cols, 1):
        m0.append(0)
        m1.append(0)

for i in range (0, rows, 1):
        if(y[i] != None and y[i] == 0):
                for j in range (0, cols, 1):
                        m0[j] += new_data[i][j]
        if(y[i] != None and y[i] == 1):
                for j in range (0, cols, 1):
                        m1[j] += new_data[i][j]

for j in range (0, cols, 1):
        m0[j]/=n[0]
        m1[j]/=n[1]

print (m0)
print (m1)

### Calculate distance of mean to each test point ###

for i in range (0, len(new_test), 1):
        d0=0
        for j in range (0,cols,1):
                d0+=(m0[j]-new_test[i][j])**2

        d1=0
        for j in range (0,cols,1):
                d1+=(m1[j]-new_test[i][j])**2

        ### Output predictions
        if(d0<d1):
                print ("0",i)
        else:
                print ("1",i)
