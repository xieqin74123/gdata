from unicodedata import name
import numpy as np
import pandas as pd
import os
from ase.io import read
from tqdm import tqdm
from bidict import bidict


def element_dic(sym):
    """
    Two way dictionary for atomic number and symbol

    Args:
        sym: atomic number or symbol. type <int> or <str>
    Returns:
        atomic_number: if input is a symbol, return atomic number.
                       type <int>
        atomic_number.inverse: if input is a number, return atomic 
                               symbol. type <str>
    """
    atomic_number = bidict({
        'H': 1,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9,
        'P': 15,
        'S': 16,
        'Cl': 17
    })
    if type(sym) == str:
        return atomic_number[sym]
    else:
        return atomic_number.inverse[sym]

class Gdata():

    def __init__(self, max_atom=100, verbose=True):
        # initialise structure, charge and name numpy array
        self.max_atom = max_atom
        self.verbose = verbose
        self.structures = np.zeros((1, self.max_atom, 4), dtype=float)
        self.charges = np.zeros((1, self.max_atom, 2), dtype=float)
        self.names = np.zeros(1, dtype=str)



    """ PRIVATE """

    # xyz reader
    def __read_xyz(self, file):
        """
        Read coordinate from .xyz file

        Args:
            file: opened xyz file. type <_io.TextIOWrapper>
        Returns:
            coordinate: structural coordinate from xyz file. type <numpy.ndarray>
        """
        # sotre current position
        position = file.tell()
        file.seek(0,0)

        # read XYZ coordinate
        file_lines=file.readlines()
        coor_temp = np.array('', dtype=str)

        for i in range(2, len(file_lines)): # skip first two lines
            line_temp = file_lines[i]
            line_temp = line_temp.strip('\n')
            coor_temp = np.append(coor_temp, line_temp)

        coor_temp = coor_temp[1:]   # drop first empty line
        df = pd.DataFrame(coor_temp)
        df = df.iloc[:, 0].str.split()
        for i in range(df.shape[0]):    # convert symbol to int
            df.iloc[i][0] = element_dic(df.iloc[i][0])
        coor_temp = np.asarray([np.stack(df.to_numpy())], dtype = float)
        coor_temp = np.pad(coor_temp, ((0,0), (0, self.max_atom-coor_temp.shape[1]), (0,0)))    # pad rest space with zero
        
        file.seek(position, 0)
        return coor_temp



    # file name extract
    def __find_real_name(self, file_name):
        """
        Get file name without path and extension

        Args:
            file_name: name of file including path and extension. type <str>
        Returns:
            pure_file_name: file name without path and extension. type <str>
        """

        # extract only file name (no dir and no file type)
        file_name_start = 0
        file_name_end = 0
        for i in range(len(file_name)):
            if file_name[i] == '/':
                file_name_start = i
            elif file_name[i] == '.':
                file_name_end = i
        if file_name_start >= file_name_end:
            file_name_end = len(file_name)

        # take file name
        real_name = file_name[file_name_start + 1:file_name_end]
        return real_name
        




    # data validation
    def __val__log(self, file):
        """
        Validate Gaussian log file if it was exit normally

        Args:
            file: opened file for validation. type <_io.TextIOWrapper>
        Returns:
            validation_result: True for passed. type <bool>
        """

        position = file.tell()  # note pointer position
        file.seek(0, 0) # move to start of file
        file_lines = file.readlines()
        lines_num = len(file_lines)
        last_line = file_lines[lines_num-1]
        find_result = last_line.find('Normal termination of Gaussian')
        file.seek(position, 0)  # move pointer back to original position
        if find_result == -1:
            return False
        else:
            return True


    
    # log reader
    def __read_log(self, file):
        """
        Read coordinate and charge from Gaussian output log file

        Args:
            file: opened log file to read. type <_io.TextIOWrapper>
        Returns:
            structure: coordinate strcture in log file. type <numpy.ndarray>
            charge: charge information in log file. type <numpy.ndarray>
        """

        # read in whole file
        position = file.tell()  # note pointer position
        file.seek(0, 0) # move to start of file
        filelines = file.readlines()
        line_max = len(filelines)

        # find coordinate and charge location in file
        for i in range(line_max):
            if (filelines[i].find('Standard orientation:') != -1):
                structure_loc = i
            if ((filelines[i].find('Mulliken charges:') != -1)
                or (filelines[i].find('Mulliken charges and spin densities:') != -1)):
                charge_loc = i
        
        # read coordinate
        structure_temp = np.array('', dtype=str)
        for i in range(structure_loc+5, line_max): # skip extra 5 lines
            line_str = filelines[i]
            if line_str.find('-----') != -1:
                break
            line_str = line_str.strip('\n') # drop '\n' at the end of line
            structure_temp = np.append(structure_temp, line_str)
        structure_temp = structure_temp[1:]   # drop first blank item in array

        # extract coordinate from string array
        df = pd.DataFrame(structure_temp)    # convert to pandas dataframe
        df = df.iloc[:,0].str.split()   # split data according to space
        structure_temp = np.asarray([np.stack(df.to_numpy())], dtype=float)   # convert back to numpy array
        structure_temp = np.delete(structure_temp, [0, 2], axis=2)    # delete useless data
        structure_temp = np.pad(structure_temp, ((0,0), (0, self.max_atom-structure_temp.shape[1]), (0,0)))    # pad rest space with zero
        
        # read mulliken charge
        charge_temp = np.array('', dtype=str)
        for i in range(charge_loc+2, line_max): # skip extra 2 lines
            line_str = filelines[i]
            if line_str.find('Sum of Mulliken charges') != -1:
                break
            line_str = line_str.strip('\n') # drop '\n' at the end of line
            charge_temp = np.append(charge_temp, line_str)
        charge_temp = charge_temp[1:]

        # extract coordiate from string array
        df = pd.DataFrame(charge_temp)
        df = df.iloc[:,0].str.split()   # split data according to space
        for i in range(df.shape[0]):    # convert element symbol to number
            df.iloc[i][1] = element_dic(df.iloc[i][1])
        charge_temp = np.asarray([np.stack(df.to_numpy())], dtype=float)   # convert back to numpy array
        charge_temp = np.delete(charge_temp, 0, axis = 2)   # delete useless data
        charge_temp = np.pad(charge_temp, ((0,0), (0, self.max_atom-charge_temp.shape[1]), (0,0))) # pad zeros

        file.seek(position, 0)  # move pointer back to original position
        return structure_temp, charge_temp



    """ PUBLIC """
    # read xyz files from dir
    def read_xyz_dir(self, dir_name):
        """
        Read all xyz files in the directory. Sotre in self.structures and self.names

        Args:
            dir_name: path/name of directory. type <str>
        Returns:
            None
        """

        # check '/'
        name_len = len(dir_name)
        if dir_name[name_len-1] != '/':
            dir_name = dir_name + '/'

        file_list = os.listdir(dir_name)        

        # verbose run
        if self.verbose == True:

            file_num = len(file_list)
            fail_num = 0
            print('Gdata: Start reading xyz data from', dir_name)            
            for file_name in tqdm(file_list):
                full_name = dir_name + file_name    # conbine name

                # skip wrong file
                try:
                    file = open(full_name, 'r')
                except:
                    print('Gdata: Fail to open file', full_name, 'Skipped!')
                    fail_num = fail_num + 1
                    continue

                try: 
                    coordinate = self.__read_xyz(file)
                except:
                    print('Gdata: Fail to read file', full_name, 'Skipped!')
                    continue
                real_name = self.__find_real_name(file_name)
                self.structures = np.append(self.structures, coordinate, axis = 0)   # append this structure to list
                self.names = np.append(self.names, real_name)
                file.close()
                
            print('Gdata:', file_num - fail_num, 'xyz files read successfully!', fail_num, 'failed')
            print('Gdata: Warning: Reading xyz files cannot obtain charge information.')

        # quiet run
        else:
            for file_name in file_list:
                full_name = dir_name + file_name

                # skip wrong file
                try:
                    file = open(full_name, 'r')
                except:
                    continue

                try:     
                    coordinate = self.__read_log(file)
                except:
                    continue
                real_name = self.__find_real_name(file_name)
                self.structures = np.append(self.structures, coordinate, axis = 0)   # append this structure to list
                self.names = np.append(self.names, real_name)
                file.close()



    # read single xyz file
    def read_xyz_file(self, file_name):
        """
        Read single xyz file and store in self.structure and self.name

        Args:
            file_name: the name of file to read. type <str>
        Returns:
            None
        """

        file = open(file_name, 'r')
        coordinate = self.__read_xyz(file)
        self.structures = np.append(self.structures, coordinate, axis = 0)

        real_name = self.__find_real_name(file_name)
        self.names = np.append(self.names, real_name, axis = 0)

        if self.verbose == True:
            print('Gdata: xyz file successfully read:', file_name)
            print('Gdata: Warning: Reading xyz files cannot obtain charge information.')
        file.close()



    # get names data
    def get_names(self):
        """
        Output stored name.

        Args:
            None
        Returns:
            names: an array of read files. type <numpy.ndarray>
        """
        return self.names[1:]



    # get charges data
    def get_charges(self):
        """
        Output stored charges data

        Args:
            None
        Returns:
            charges: an array contains charges distribution. type <numpy.ndarray>
        """
        return self.charges[1:]



    # get structure data
    def get_structures(self):
        """
        Output stored structures data

        Args:
            None
        Returns:
            structures: an array contains structural coordinates. type <numpy.ndarray>
        """
        return self.structures[1:]



    # load data from npy
    def load(self, structure_name='structures.npy', charge_name='charges.npy', name_name='names.npy'):
        """
        Load saved structure, charge and name data from .npy file

        Args:
            structure_name: path and name of saved structure data. type <str>
            charge_name: path and name of saved charge data. type <str>
            name_name: path and name of saved charge data. type <str>
        Returns:
            None
        """
        structures = np.load(structure_name)
        charges = np.load(charge_name)
        # check data size
        if structures.shape[0] != charges.shape[0]:
            print('Gdata: Number of data does not match each other! Loading aborted!')
            print('Gdata: Number of structure data:', structures.shape[0])
            print('Gdata: Number of charge data:', charges.shape[0])
            exit()
        elif structures.shape[2] != 4 or charges.shape[2] != 2:
            print('Gdata: Illegal data shape! Loading aborted!')
            print('Gdata: Shape of structure data:', structures.shape)
            print('Gdata: Shape of charge data:', charges.shape)
            exit()
        elif structures.shape[1] != charges.shape[1]:
            print('Gdata: Maximum allowed atom does not match each other! Loading aborted!')
            print('Gdata: Maximum allowed atom of structure data:', structures.shape[1])
            print('Gdata: Maximum allowed atom of charge data:', charges.shape[1])
            exit()
        else:
            # modified preset size according to data
            self.max_atom = charges.shape[1]
            self.structures = np.zeros((1, self.max_atom, 4), dtype=float)
            self.charges = np.zeros((1, self.max_atom, 2), dtype=float)
            # mount data to class
            self.structures = np.append(self.structures, structures, axis=0)
            self.charges = np.append(self.charges, charges, axis=0)

            if self.verbose == True:
                print('Gdata:',charges.shape[0], 'data loaded and validated. \
                    Maximum allowed atom changed to', self.max_atom)
        



    # save data as npy
    def save(self, structure_name='structures.npy', charge_name='charges.npy', name_name='name.npy'):
        """
        Save data as numpy .npy data

        Args:
            structure_name: path and name of structure data. type <str>
            charge_name: path and name of charge data. type <str>
            name_name: path and name of name data. type <str>
        Returns:
            None
        """
        
        # exclude first zero item
        structures = self.structures[1:]
        charges = self.charges[1:]
        names = self.names[1:]
        np.save(structure_name, structures)
        np.save(charge_name, charges)
        np.save(name_name, names)
        if self.verbose == True:
            print('Gdata: Data saved as', structure_name, ',', charge_name, 'and', name_name)
        



    # read files in dir
    def read_log_dir(self, dir_name, validation=True):
        """
        Read Gaussian log files from directory and store in self.structure, 
        self.charges and self.name

        Args:
            dir_name: path and name of directory. type <str>
            validation: enable validation for log, this will check whether 
                        the Gaussian calculation exit normally in this log 
                        file. type <bool>
        Returns:
            None
        """

        # check '/'
        name_len = len(dir_name)
        if dir_name[name_len-1] != '/':
            dir_name = dir_name + '/'
        
        file_list = os.listdir(dir_name)   
        
        # verbose run
        if self.verbose == True:

            file_num = len(file_list)
            fail_num = 0   
            print('Gdata: Start reading log data from', dir_name)            
            for file_name in tqdm(file_list):
                full_name = dir_name + file_name    # conbine name

                # skip wrong file
                try:
                    file = open(full_name, 'r')
                except:
                    print('Gdata: Fail to open file', full_name, 'Skipped!')
                    fail_num = fail_num + 1
                    continue

                # validate file and skip bad file
                if validation == True:
                    if self.__val__log(file) == False:
                        print('Gdata: file', file_name, 'did not pass validation! Skipped!')
                        fail_num = fail_num + 1
                        file.close()
                        continue

                try:
                    structure, charge = self.__read_log(file)
                except:
                    print('Gdata: Fail to read file', full_name, 'Skipped!')
                    continue
                real_name = self.__find_real_name(file_name)
                self.structures = np.append(self.structures, structure, axis = 0)   # append this structure to list
                self.charges = np.append(self.charges, charge, axis = 0) # append this charge to list
                self.names = np.append(self.names, real_name)
                file.close()
                
            print('Gdata:', file_num - fail_num, 'log files read successfully!', fail_num, 'failed')

        # quiet run
        else:
            for file_name in file_list:
                full_name = dir_name + file_name

                # skip wrong file
                try:
                    file = open(full_name, 'r')
                except:
                    continue

                # validate file and skip bad file
                if validation == True:
                    if self.__val__log(file) == False:
                        continue

                structure, charge = self.__read_log(file)
                real_name = self.__find_real_name(file_name)
                self.structures = np.append(self.structures, structure, axis = 0)   # append this structure to list
                self.charges = np.append(self.charges, charge, axis = 0) # append this charge to list
                self.names = np.append(self.names, real_name)
                file.close()




    # read single file
    def read_log_file(self, file_name, validation=True):
        """
        Read single Gaussian log file store in self.structure, 
        self.charges and self.name

        Args:
            file_name: path and name of Gaussian log file. type <str>
            validation: enable validation for log, this will check whether 
                        the Gaussian calculation exit normally in this log 
                        file. type <bool>
        Returns:
            None
        """
        # open file
        file = open(file_name, 'r')
        # validate file
        if validation == True:
            if self.__val__log(file) == False:
                print('Gdata: file', file_name, 'did not pass validation! Aborted!')
                file.close()
                exit()
        
        # read data from file
        structure, charge = self.__read_log(file)
        real_name = self.__find_real_name(file_name)
        self.structures = np.append(self.structures, structure, axis = 0)   # append this structure to list
        self.charges = np.append(self.charges, charge, axis = 0) # append this charge to list
        self.names = np.append(self.names, real_name)
        file.close()

        # verbose info
        if self.verbose == True:
            print('Gdata: log file successfully read:', file_name)