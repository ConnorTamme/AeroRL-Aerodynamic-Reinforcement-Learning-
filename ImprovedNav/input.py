open("input.txt", "w").close()
while(True):
    cmd = input("Enter input:")
    file = open("input.txt", "w")
    file.write(cmd)
    file.close()
