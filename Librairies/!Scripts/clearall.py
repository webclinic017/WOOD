def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]
        
clearall()