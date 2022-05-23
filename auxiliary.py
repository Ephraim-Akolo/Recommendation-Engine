
def write_file(value:list, file_name="test_file.txt", path=".", new=False):
    '''
    writes data to text file.
    
    parameters:

        value (list): list of objects to write to text file.

        file_name (str): name of file to write to.

    '''
    import numpy
    from pathlib import Path
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)
    path = Path(path)/file_name
    for val in value:
        if not isinstance(val, str):
            val = str(val)
        if new:
            new = False
            with open(path, 'w') as f:
                f.write(val+"\n\n")
        else:
            try:
                with open(path, 'a',) as f:
                    f.write(val+"\n\n")
            except FileNotFoundError:
                with open(path, 'w') as f:
                    f.write(val+"\n\n")
    numpy.set_printoptions(**opt)
