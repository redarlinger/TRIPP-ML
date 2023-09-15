from PIL import Image

def transient_positions(filepath): #filepath input is a string
	path1 = '/mnt/annex/ryan/LCO/LCO_set8_pngs/' #this is where the pngs are located
	path2 = '/mnt/annex/ryan/set_8/lists/' #this is where the differences.txt files are located
	ans = [] #this function will eventually return a list of lists, with an x and y value corresponding to each of the transients in the image [[x1,y1],[x2,y2]...]	
	temp = [] #a temporary variable used to store variables until they are manipulated and put into ans
	width, height = Image.open(path1 + filepath + '/0.fits.png').size #get the size of the image in pixels
	
	#now we need to open the corresponding differences.txt file to the png	
	f = open(path2 + filepath + '/differences.txt')
	for i in f: #this block of code isolates the x and y positions of the transients in each png
		x0 = list(i.split())	
		x1 = x0[2].replace("'","")
		x1 = x1.replace(',','')	
		x1 = float(x1)	
		x2 = x0[3].replace("'","")	
		x2 = x2.replace(',','')	
		x2 = float(x2)	
		temp.append([x1,x2])
	
	for i in temp: #now we take the x and y positions of each of the transients and position them relative to the size of the image
		j1 = i[0]
		j2 = i[1]
		x = j1/width
		y = 1 - j2/height
		ans.append([x,y])
	f.close()
	return(ans)
