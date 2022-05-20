
def write_file(value:list, file_name="test_file.txt", dir=".", new=False):
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)
    for val in value:
        if not isinstance(val, str):
            val = str(val)
        if new:
            new = False
            with open(file_name, 'w') as f:
                f.write(val+"\n\n")
        else:
            try:
                with open(file_name, 'a') as f:
                    f.write(val+"\n")
            except FileNotFoundError:
                with open(file_name, 'w') as f:
                    f.write(val+"\n")
    numpy.set_printoptions(**opt)
