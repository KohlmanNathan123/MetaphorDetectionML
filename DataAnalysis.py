import csv

# counts how many metaphor examples there are
def metaphor_count(text):
    with open(text, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        count = 0

    for row in rows:
        if(row[1]=='TRUE'):
            count += 1
    return count

# counts how many examples exist of metaphor and non-metaphor for each word 
def count_categories(text):
    with open(text, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        road = 0
        r_metaphor = 0
        candle = 0
        c_metaphor = 0
        light = 0
        l_metaphor = 0
        spice = 0
        s_metaphor = 0
        ride = 0
        ri_metaphor = 0
        train = 0
        t_metaphor = 0
        boat = 0
        b_metaphor = 0

    for row in rows:
        if(row[0]=='0'):
            road += 1
            if(row[1]=='TRUE'):
                r_metaphor += 1
        elif(row[0]=='1'):
            candle += 1
            if(row[1]=='TRUE'):
                c_metaphor += 1
        elif(row[0]=='2'):
            light += 1
            if(row[1]=='TRUE'):
                l_metaphor += 1
        elif(row[0]=='3'):
            spice += 1
            if(row[1]=='TRUE'):
                s_metaphor += 1
        elif(row[0]=='4'):
            ride += 1
            if(row[1]=='TRUE'):
                ri_metaphor += 1
        elif(row[0]=='5'):
            train += 1
            if(row[1]=='TRUE'):
                t_metaphor += 1
        else:
            boat += 1
            if(row[1]=='TRUE'):
                b_metaphor += 1

        categories = [road, candle, light, spice, ride, train, boat]
        metaphors = [r_metaphor,c_metaphor,l_metaphor,s_metaphor,ri_metaphor,t_metaphor,b_metaphor]
        not_metaphors = [road - r_metaphor,candle - c_metaphor,light - l_metaphor,spice - s_metaphor,ride - ri_metaphor,train - t_metaphor,boat - b_metaphor]

    return categories, metaphors, not_metaphors

# calculates percent
def percent(part, total):
    rhs = part * 100
    percent = rhs/total
    return round(percent)

# calculates percent for each element in a list
def list_percent(part, total):
    for p in range(len(part)):
        part[p] = part[p]*100
    percent = []
    for x in range(len(part)):
        result = part[x]/total[x]
        percent.append(round(result))
    return percent

if __name__ == "__main__":
    metaphors = metaphor_count('train.csv')
    print("Number of Metaphors: " , metaphors)
    print("Number of Total Examples: 1870")
    percent_metaphors = percent(metaphors,1870)
    print("Percent metaphors:  " , percent_metaphors ,"\n")


    print("Words: road, candle, light, spice, ride, train, boat\n")
    total, metaphor, not_metaphor = count_categories('train.csv')
    print("Total for each word: ", total)
    print("Metaphors for each word: ", metaphor)
    print("Not Metaphors for each word: ", not_metaphor, "\n")

    percent_metaphor = list_percent(metaphor,total)
    print("Percentage Metaphor: ", percent_metaphor)
    percent_not_metaphor = list_percent(not_metaphor,total)
    print("Percentage Not Metaphor: ", percent_not_metaphor)
