from io import TextIOWrapper
from math import floor
import numpy as np
import os
from tqdm import tqdm
from bidict import bidict
# Gdata version
version = 'master'


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

    def __init__(self, charge_type='Mulliken', max_atom=100):
        """
        Gaussian data class.

        Args:
            max_atom: maximum allowed atoms in single molecule. type <int>
        """
        # initialise structure, charge and name numpy array
        self.max_atom = max_atom
        self.charge_type=charge_type
        self.structures = np.zeros((1, self.max_atom, 4), dtype=float)
        self.charges = np.zeros((1, self.max_atom), dtype=float)
        self.names = np.zeros(1, dtype=str)
        self.topologies = np.zeros(
            (1, self.max_atom, self.max_atom), dtype=int)
        self.dipoles = np.zeros((1, 3), dtype=float)

    """ PRIVATE """

    def __data_check(self,
                     structures=None,
                     charges=None,
                     names=None,
                     topologies=None,
                     dipoles=None) -> bool:
        """
        Check input data whether they match each other and the requirement

        Args:
            structures: array of structure data. type <np.ndarray>
            charges: array of charge data. type <np.ndarray>
            names: array of name data. type <np.ndarray>
            topologies: array of topology data. type <np.ndarray>
            dipoles: array of dipole moment data. type <np.ndarray>
        Returns:
            check_result: if data pass check, return True. type <bool>
        """

        data_num = []
        max_atom = []

        if structures is not None and structures.shape[0] != 0:
            if structures.ndim != 3:
                print(
                    'Gdata: loaded structures data dimension did not match requirement!')
                return False
            if structures.shape[2] != 4:
                print('Gdata: loaded structures data shape did not match requirement!')
                return False
            data_num.append(structures.shape[0])
            max_atom.append(structures.shape[1])

        if charges is not None and charges.shape[0] != 0:
            if charges.ndim != 2:
                print('Gdata: loaded charges data dimension did not match requirement!')
                return False
            data_num.append(charges.shape[0])
            max_atom.append(charges.shape[1])

        if names is not None and names.shape[0] != 0:
            if names.ndim != 1:
                print('Gdata: loaded names data dimension did not match requirement!')
                return False
            data_num.append(names.shape[0])

        if topologies is not None and topologies.shape[0] != 0:
            if topologies.ndim != 3:
                print(
                    'Gdata: loaded topologies data dimension did not match requirement!')
                return False
            if topologies.shape[1] != topologies.shape[2]:
                print('Gdata: loaded topologies data shape did not symmetric')
                return False
            data_num.append(topologies.shape[0])
            max_atom.append(topologies.shape[1])

        if dipoles is not None and dipoles.shape[0] != 0:
            if dipoles.ndim != 2:
                print('Gdata: loaded names data dimension did not match requirement!')
                return False
            data_num.append(dipoles.shape[0])

        # check whether data have same data number and max_atom
        if data_num.count(data_num[0]) != len(data_num):
            print('Gdata: the number of data does not match between data!')
            return False
        if max_atom.count(max_atom[0]) != len(max_atom):
            print('Gdata: the maximum allowed atom number does not match between data!')
            return False

        return True

    def __read_zmat(self, file: TextIOWrapper) -> np.ndarray:
        """
        read in .com file generate by Gaussian newzmat

        Args:
            file: opened file to read in. type <TextIOWrapper>
        Returns:
            coordinate: xyz coordinate of molecule. type <numpy.ndarray>
            topologies: topological information of bonding. type <numpy.ndarray>
        """
        # store current position
        position = file.tell()
        file.seek(0, 0)

        # read in data
        file_lines = file.readlines()
        lines_num = len(file_lines)

        coordinate_start = 0

        # find coordinate start and stop
        for i in range(lines_num):
            if file_lines[i].find('No Title Specified') != -1:
                coordinate_start = i + 3    # skip extra 3 lines
                break

        # if no coordinate found
        if coordinate_start == 0:
            raise

        coordinate_end = coordinate_start
        for i in range(coordinate_start, lines_num):
            if file_lines[i] == '\n':
                coordinate_end = i
                break

        # read in coordinate
        # example of line:
        #   C                                                -2.725769000000     -0.395935000000     -0.210123000000
        coor_temp = np.zeros((1, 4), dtype=float)
        for i in range(coordinate_start, coordinate_end):
            line_temp = file_lines[i].split()
            # convert element sym to num
            line_temp[0] = element_dic(line_temp[0])
            line_temp = np.array(line_temp, dtype=float)
            coor_temp = np.append(coor_temp, [line_temp], axis=0)

        coor_temp = coor_temp[1:]
        coor_temp = np.pad(
            coor_temp, ((0, self.max_atom-coor_temp.shape[0]), (0, 0)))

        # read in topologies
        # example of line:
        #  1 2 1.000 8 1.000 9 1.000 10 1.000
        topologies_start = coordinate_end + 1
        topo_temp = np.zeros((self.max_atom, self.max_atom), dtype=int)
        for i in range(topologies_start, lines_num):
            line_temp = file_lines[i]
            if line_temp == ' \n':
                break
            line_temp = line_temp.split()
            atom1 = int(line_temp[0]) - 1
            topo_temp[atom1, atom1] = 0    # set self bond to 0
            for j in range(1, len(line_temp), 2):
                atom2 = int(line_temp[j]) - 1
                bonding = int(floor(float(line_temp[j+1])))
                topo_temp[atom1, atom2] = bonding
                topo_temp[atom2, atom1] = bonding
        # move back pointer
        file.seek(position, 0)
        return coor_temp, topo_temp

    # write xyz file
    def __write_xyz(self, file: TextIOWrapper, coordinate: np.ndarray) -> int:
        """
        write xyz coordinate to file and return actual number of atom in molecule

        Args:
            file: file to write. type <_io.TextIOWrapper>
            coordinate: coordinate of structure in shape [max_atom, 4]. type <numpy.ndarray>
        Returns:
            atom_num: actual number of atom in this structure. type <int>
        """

        # print lines
        for i in range(self.max_atom):
            # stop while meet padded zero
            if coordinate[i][0] == 0:
                atom_num = i
                break
            else:
                # print elements
                print('%c %f %f %f' % (element_dic(
                    coordinate[i][0]), coordinate[i][1], coordinate[i][2], coordinate[i][3]), file=file)
        return atom_num

    # xyz reader

    def __read_xyz(self, file: TextIOWrapper, header=True) -> np.ndarray:
        """
        Read coordinate from .xyz file

        Args:
            file: opened xyz file. type <_io.TextIOWrapper>
        Returns:
            coordinate: structural coordinate from xyz file. type <numpy.ndarray>
        """
        # store current position
        position = file.tell()
        file.seek(0, 0)

        file_lines = file.readlines()

        if header is True:
            start_line = 2
        else:
            start_line = 0

        # read XYZ coordinate
        coor_temp = np.zeros((1, 4), dtype=float)
        for i in range(start_line, len(file_lines)):  # skip first two lines
            line_temp = file_lines[i].split()
            # convert element sym to num
            line_temp[0] = element_dic(line_temp[0])
            line_temp = np.array(line_temp, dtype=float)
            coor_temp = np.append(coor_temp, [line_temp], axis=0)

        coor_temp = coor_temp[1:]
        # print(coor_temp)
        # pad rest space with zero
        coor_temp = np.pad(
            coor_temp, ((0, self.max_atom-coor_temp.shape[0]), (0, 0)))

        file.seek(position, 0)
        return coor_temp

    # file name extract

    def __find_real_name(self, file_name: str) -> str:
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

        # bug_fix: no skip if name has no path
        if file_name_start != 0:
            file_name_start = file_name_start + 1

        # take file name
        real_name = file_name[file_name_start:file_name_end]
        return real_name

    # data validation

    def __val__log(self, file: TextIOWrapper) -> bool:
        """
        Validate Gaussian log file if it was exit normally

        Args:
            file: opened file for validation. type <_io.TextIOWrapper>
        Returns:
            validation_result: True for passed. type <bool>
        """

        position = file.tell()  # note pointer position
        file.seek(0, 0)  # move to start of file
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

    def __read_log(self, file: TextIOWrapper) -> np.ndarray:
        """
        Read coordinate and charge from Gaussian output log file

        Args:
            file: opened log file to read. type <_io.TextIOWrapper>
        Returns:
            structure: coordinate strcture in log file. type <numpy.ndarray>
            charge: charge information in log file. type <numpy.ndarray>
            dipole_moment: dipole moment in log file. type <float>
        """

        # read in whole file
        position = file.tell()  # note pointer position
        file.seek(0, 0)  # move to start of file
        filelines = file.readlines()
        line_max = len(filelines)
        
        structure_loc = None
        Mcharge_loc = None
        dipole_loc = None
        Hcharge_loc = None

        # find coordinate and charge location in file
        for i in range(line_max):
            if (filelines[i].find('Standard orientation:') != -1):
                structure_loc = i
            if ((filelines[i].find('Mulliken charges:') != -1)
                    or (filelines[i].find('Mulliken charges and spin densities:') != -1)):
                Mcharge_loc = i
            if (filelines[i].find('Dipole moment (field-independent basis, Debye):') != -1):
                dipole_loc = i
            if (filelines[i].find('Hirshfeld charges, spin densities, dipoles, and CM5 charges') != -1):
                Hcharge_loc = i

        # read coordinate
        structure_temp = np.zeros((1, 4), dtype=float)
        for i in range(structure_loc+5, line_max):  # skip extra 5 lines
            line_temp = filelines[i]
            if line_temp.find('-----') != -1:
                break
            line_temp = line_temp.split()
            line_temp = np.array(line_temp, dtype=float)
            line_temp = np.delete(line_temp, [0, 2])    # delete useless data
            structure_temp = np.append(structure_temp, [line_temp], axis=0)
        structure_temp = structure_temp[1:]   # drop first blank item in array
        structure_temp = np.pad(structure_temp, ((
            0, self.max_atom-structure_temp.shape[0]), (0, 0)))    # pad rest space with zero

        # read mulliken charge
        if self.charge_type == 'Mulliken':
            charge_temp = np.zeros(1, dtype=float)
            for i in range(Mcharge_loc+2, line_max):  # skip extra 2 lines
                line_temp = filelines[i]
                if line_temp.find('Sum of Mulliken charges') != -1:
                    break
                line_temp = line_temp.split()
                line_temp[1] = element_dic(line_temp[1])
                line_temp = np.array(line_temp, dtype=float)
                line_temp = np.delete(line_temp, [0, 1])   # delete useless data
                charge_temp = np.append(charge_temp, line_temp, axis=0)
            charge_temp = charge_temp[1:]
            charge_temp = np.pad(
                charge_temp, (0, self.max_atom-charge_temp.shape[0]))  # pad zeros

        # read hirshfeld charge
        elif self.charge_type == 'Hirshfeld':
            if Hcharge_loc is None:
                raise ValueError('No Hirshfeld charge founded here!')
            charge_temp = np.zeros(1, dtype=float)
            for i in range(Hcharge_loc+2, line_max):
                line_temp = filelines[i]
                if line_temp.find('Tot') != -1:
                    break
                line_temp = line_temp.split()
                line_temp[1] = element_dic(line_temp[1])
                line_temp = np.array(line_temp, dtype=float)
                line_temp = line_temp[2, np.newaxis]
                charge_temp = np.append(charge_temp, line_temp, axis=0)
            charge_temp = charge_temp[1:]
            charge_temp = np.pad(
                charge_temp, (0, self.max_atom-charge_temp.shape[0]))  # pad zeros

        # read dipole moment
        line_temp = filelines[dipole_loc+1]
        line_temp = line_temp.split()
        dipole_temp = []
        dipole_temp.append(float(line_temp[1]))
        dipole_temp.append(float(line_temp[3]))
        dipole_temp.append(float(line_temp[5]))
        dipole_temp = np.array(dipole_temp, dtype=float)

        file.seek(position, 0)  # move pointer back to original position
        return structure_temp, charge_temp, dipole_temp

    """ PUBLIC """

    def add_data(self,
                 structure: np.ndarray = None,
                 charge: np.ndarray = None,
                 name: np.ndarray = None,
                 topology: np.ndarray = None,
                 dipole: np.ndarray = None):
        """
        Add single data to class

        Args:
            max_atom: maximum allowed atom in each structure. type <int>
            structures: array of structural coordinates. type <numpy.ndarray>
            charges: array of charges distribution. type <numpy.ndarray>
            names: array of names of structures. type <numpy.ndarray>
            topologies: array of topological information. type <numpy.ndarray>
        Returns:
            None
        """

        if type(structure) == np.ndarray:
            if structure.ndim == 2:
                self.structures = np.append(
                    self.structures, [structure], axis=0)
            elif structure.ndim == 3:
                self.structures = np.append(self.structures, structure, axis=0)
            else:
                print('Gdata: structure data dimension error!')
                raise
        if type(charge) == np.ndarray:
            if charge.ndim == 1:
                self.charges = np.append(self.charges, [charge], axis=0)
            elif charge.ndim == 2:
                self.charges = np.append(self.charges, charge, axis=0)
            else:
                print('Gdata: charge data dimension error!')
                raise
        if type(name) != type(None):
            self.names = np.append(self.names, name)
        if type(topology) == np.ndarray:
            if topology.ndim == 2:
                self.topologies = np.append(
                    self.topologies, [topology], axis=0)
            elif topology.ndim == 3:
                self.topologies = np.append(self.topologies, topology, axis=0)
            else:
                print('Gdata: topology data dimension error!')
                raise
        if type(dipole) == np.ndarray:
            if dipole.ndim == 1:
                self.dipoles = np.append(self.dipoles, [dipole], axis=0)
            elif dipole.ndim == 2:
                self.dipoles = np.append(self.dipoles, dipole, axis=0)
            else:
                print('Gdata: dipole moment data dimension error!') 

    def pad_zeros(self):
        """
        Pad zeros to thoes blank without data to fit in shape

        Args:
            None
        Returns:
            None
        """

        data_num = self.get_data_shape()[0] + 1

        self.structures = np.pad(
            self.structures, ((0, data_num-self.structures.shape[0]), (0, 0), (0, 0)))
        self.charges = np.pad(
            self.charges, ((0, data_num-self.charges.shape[0]), (0, 0)))
        self.names = np.pad(self.names, (0, data_num-self.names.shape[0]))
        self.topologies = np.pad(
            self.topologies, ((0, data_num-self.topologies.shape[0]), (0, 0), (0, 0)))
        self.dipoles = np.pad(self.dipoles, ((0, data_num-self.dipoles.shape[0]), (0, 0)))

    def delete_dipole(self):
        """
        Delete all dipole moment info in class

        Args:
            None
        Return:
            None
        """
        self.dipoles = np.zeros((1, 3), dtype=float)

    def delete_topologies(self):
        """
        Delete all topology data in calss

        Args:
            None
        Returns:
            None
        """
        self.topologies = np.zeros(
            (1, self.max_atom, self.max_atom), dtype=int)

    def delete_names(self):
        """
        Delete all name data in calss

        Args:
            None
        Returns:
            None
        """
        self.names = np.zeros(1, dtype=str)

    def delete_charges(self):
        """
        Delete all charge data in class

        Args:
            None
        Returns:
            None
        """
        self.charges = np.zeros((1, self.max_atom), dtype=float)

    def delete_structures(self):
        """
        Delete all structure data in class

        Args:
            None
        Returns:
            None
        """
        self.structures = np.zeros((1, self.max_atom, 4), dtype=float)

    def change_max_atom(self, new_max_atom: int):
        """
        Change max_atom number in class

        Args:
            new_max_atom: desired maximum allowed atom in molecule. type <int>
        Returns:
            None
        """

        # minimise first
        self.minimise()

        if new_max_atom < self.max_atom:
            print(
                'Gdata: assigned maximum allowed number of atom is too small to hold all data!')
            raise

        pad_num = new_max_atom - self.max_atom

        self.structures = np.pad(
            self.structures, ((0, 0), (0, pad_num), (0, 0)))
        self.charges = np.pad(self.charges, ((0, 0), (0, pad_num)))
        self.topologies = np.pad(
            self.topologies, ((0, 0), (0, pad_num), (0, pad_num)))

    def self_check(self) -> bool:
        """
        Check stored data whether valid or not

        Args:
            None
        Return:
            check_result: if True, data has passed the check
        """
        structures = None
        charges = None
        names = None
        topologies = None
        dipoles = None

        if self.structures.shape[0] != 1:
            structures = self.structures[1:]
        if self.charges.shape[0] != 1:
            charges = self.charges[1:]
        if self.names.shape[0] != 1:
            names = self.names[1:]
        if self.topologies.shape[0] != 1:
            topologies = self.topologies[1:]
        if self.dipoles.shape[0] != 1:
            dipoles = self.dipoles[1:]

        return self.__data_check(structures, charges, names, topologies, dipoles)

    # read zmat files from directory

    def read_zmat_dir(self, dir_name: str):
        """
        Read all zmat files from directory

        Args:
            dir_name: path and name of the directory. type <str>
        Returns:
            None
        """
        # check '/'
        name_len = len(dir_name)
        if dir_name[name_len-1] != '/':
            dir_name = dir_name + '/'

        file_list = os.listdir(dir_name)

        file_num = len(file_list)
        io_error_names = []
        content_error_names = []

        print('Gdata: Start reading zmat data from %s' % (dir_name))
        for file_name in tqdm(file_list):
            full_name = dir_name + file_name    # conbine name

            # skip wrong file
            try:
                file = open(full_name, 'r')
            except:
                io_error_names.append(full_name)
                continue

            try:
                coordinate, topology = self.__read_zmat(file)
            except:
                content_error_names.append(full_name)
                continue

            real_name = self.__find_real_name(file_name)
            # append this structure to list
            self.structures = np.append(
                self.structures, [coordinate], axis=0)
            self.topologies = np.append(
                self.topologies, [topology], axis=0)
            self.names = np.append(self.names, real_name)
            file.close()

        io_error_num = len(io_error_names)
        content_error_num = len(content_error_names)
        error_num = io_error_num + content_error_num

        print('Gdata: %d zmat files read successfully! %d failed!' %
              (file_num-error_num, error_num))

        if io_error_num != 0:
            print('Gdata: %d operations failed due to IO errors, which are: \n %s.' % (
                io_error_num, io_error_names))
        if content_error_num != 0:
            print('Gdata: %d operations failed due to invalid contents in input files, which are: \n %s.' % (
                content_error_num, content_error_names))

    # read single zmat file

    def read_zmat_file(self, file_name: str):
        """
        Read single newzmat .com file

        Args:
            file_name: path and name of file. type <str>
        Return:
            None
        """
        try:
            file = open(file_name, 'r')
        except:
            print('Gdata: operation failed due to IO errors')
            raise

        try:
            structure, topology = self.__read_zmat(file)
        except:
            print('Gdata: operation failed due to invalid contents in input file.')
            raise

        real_name = self.__find_real_name(file_name)

        # append this structure to list
        self.structures = np.append(self.structures, [structure], axis=0)
        self.names = np.append(self.names, real_name)
        self.topologies = np.append(self.topologies, [topology], axis=0)
        file.close()

        print('Gdata: zmat file read successfully: %s' % file_name)

    # minimise max_atom number

    def minimise(self):
        """
        delete redundant '0' atoms to minimise the shape of array

        Args:
            None
        Return:
            None
        """

        max_atom_num = 0
        data_num = self.get_data_shape()[0]

        # a blank element is exist in self.structure[0]
        for i in range(1, data_num + 1):
            for j in range(0, self.max_atom):
                atom_info = self.structures[i, j]
                if atom_info[0] == 0:
                    if j + 1 > max_atom_num:
                        max_atom_num = j + 1
                    break
        try:
            self.structures = self.structures[:, :max_atom_num, :]
            self.charges = self.charges[:, :max_atom_num]
            self.topologies = self.topologies[:, :max_atom_num, :max_atom_num]
        finally:
            self.max_atom = max_atom_num
            print('Gdata: number of maximum allowed atom has been minimised to %d' % (max_atom_num))

    # convert to xyz

    def convert_to_xyz(self, directory: str, header=True):
        """
        Convert stored structure data to xyz format

        Args:
            directory: directory of output files. type <str>
            header: if True, print .xyz header. type <bool>
        Returns:
            None
        """

        # check '/'
        name_len = len(directory)
        if directory[name_len-1] != '/':
            directory = directory + '/'

        # check dir exist
        if os.path.exists(directory) == False:
            os.mkdir(directory)

        structures = self.structures[1:]
        names = self.names[1:]

        data_num = self.get_data_shape()[0]

        fail_names = []

        # write xyz
        print('Gdata: Start writing xyz files...')
        for i in tqdm(range(data_num)):
            file_name = names[i] + '.xyz'
            full_name = directory + file_name
            try:
                file = open(full_name, 'w+')
            except:
                fail_names.append(file_name)
                continue

            atom_num = self.__write_xyz(file, structures[i])

            # print header
            if header == True:
                file.seek(0, 0)
                file_lines = file.readlines()
                header_line = str(atom_num) + '\n\n'
                file_lines.insert(0, header_line)
                file.seek(0, 0)
                file.writelines(file_lines)
                file.close()

        fail_num = len(fail_names)
        print('Gdata: %d xyz files writed successfully!' % (data_num-fail_num))
        if fail_num != 0:
            print('Gdata: %d operations failed due to IO errors, which are: \n%s' % (
                fail_num, fail_names))

    def get_data_shape(self) -> np.ndarray:
        """
        Get number of data stored in class, for structures, charges and names

        Args:
            None
        Return:
            shape_array: an array of shape in order of [structures, charges, names, topologies].
                         type <numpy.ndarray>
        """

        structure_shape = self.structures.shape[0] - 1
        charge_shape = self.charges.shape[0] - 1
        name_shape = self.names.shape[0] - 1
        topology_shape = self.topologies.shape[0] - 1

        shape_array = np.array(
            [structure_shape, charge_shape, name_shape, topology_shape], dtype=int)

        return shape_array

    # read xyz files from dir

    def read_xyz_dir(self, dir_name: str, header=True):
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
                coordinate = self.__read_xyz(file, header)
            except:
                print('Gdata: Fail to read file', full_name, 'Skipped!')
                continue
            real_name = self.__find_real_name(file_name)
            # append this structure to list
            self.structures = np.append(
                self.structures, [coordinate], axis=0)
            self.names = np.append(self.names, real_name)
            file.close()

        print('Gdata:', file_num - fail_num,
                'xyz files read successfully!', fail_num, 'failed')
        print('Gdata: Warning: Reading xyz files cannot obtain charge information.')

    # read single xyz file

    def read_xyz_file(self, file_name: str, header=True):
        """
        Read single xyz file and store in self.structure and self.name

        Args:
            file_name: the name of file to read. type <str>
        Returns:
            None
        """

        file = open(file_name, 'r')
        coordinate = self.__read_xyz(file, header)
        self.structures = np.append(self.structures, [coordinate], axis=0)

        real_name = self.__find_real_name(file_name)
        self.names = np.append(self.names, real_name)

        print('Gdata: xyz file successfully read:', file_name)
        print('Gdata: Warning: Reading xyz files cannot obtain charge information.')
        file.close()

    # get dipole moment info
    def get_dipole(self, style:str='norm') -> np.ndarray:
        """
        Output dipole moment

        Args:
            style: select output style. <str>
                    norm: output the norm of dipole moment
                    xyz: output dipole in x, y and z directions
                    ivec:
        Return:
            dipole: dipole moment information. type <numpy.ndarray>
        """
        if style == 'xyz':
            return self.dipoles[1:]
        elif style == 'norm':
            return np.linalg.norm(self.dipoles[1:], axis=1)

    # get degree matrix
    def get_degree(self) -> np.ndarray:
        """
        Output degree matrix for GCN

        Args:
            None
        Returns:
            degree: degree matrix for GCN. type <numpy.ndarray>
        """

        # modified topologies to degree matrix
        adjacency = self.get_adjacency()
        degree = np.zeros(adjacency.shape)

        for data_num in range(adjacency.shape[0]):
            for rowcol in range(adjacency.shape[1]):
                degree[data_num, rowcol, rowcol] = np.sum(
                    adjacency[data_num, rowcol])

        return degree

    # get adjacency matrix
    def get_adjacency(self, self_loop: bool = False) -> np.ndarray:
        """
        Output adjacency for GCN

        Args:
            self_loop: add self-bond to adjacency matrix. type <bool>
        Returns:
            adjacency: adjacency matrix for GCN. type <numpy.ndarray>
        """

        # modified topologies to ajacency matrix, move all non-zero to 1
        adjacency = self.topologies[1:]

        for data_num in range(adjacency.shape[0]):
            for row in range(adjacency.shape[1]):
                for col in range(adjacency.shape[2]):
                    if adjacency[data_num, row, col] == 0:
                        continue
                    adjacency[data_num, row, col] = 1

        # add identity
        if self_loop == True:
            identity = np.zeros(adjacency.shape)
            for data_num in range(adjacency.shape[0]):
                for rowcol in range(adjacency.shape[1]):
                    if np.sum(adjacency[data_num, rowcol]) != 0:
                        identity[data_num, rowcol, rowcol] = 1
            adjacency = adjacency + identity

        return adjacency

    # get topological data
    def get_atom_info(self, format: str = 'array') -> np.ndarray:
        """
        Output atom type in order

        Args:
            format: output format. type <str>
                'array': default value. output atomic number in array
                'matrix': output atomic number in matrix
        Returns:
            atom_info: atomic number in array or matrix. type <numpy.ndarray>
        """

        atom_info = np.array(self.structures[1:, :, 0], dtype=int)

        if format == 'matrix':
            atom_matrix = np.zeros(
                (1, self.max_atom, self.max_atom), dtype=int)
            data_num = self.get_data_shape().max()
            print('Gdata: reshaping atomic info...')
            for i in tqdm(range(data_num)):
                atom_matrix_temp = np.zeros(
                    (self.max_atom, self.max_atom), dtype=int)
                for j in range(self.max_atom):
                    atom_matrix_temp[j][j] = atom_info[i][j]
                atom_matrix = np.append(
                    atom_matrix, [atom_matrix_temp], axis=0)

            return atom_matrix[1:]

        elif format == 'array':
            return atom_info

    def get_topologies(self, self_loop: bool = False) -> np.ndarray:
        """
        Output stored topological data

        Args:
            self_loop: add self loop to topology matrix. type <bool>
        Returns:
            topologies: and matrix of topologies info. type <numpy.ndarray>
        """
        topologies = self.topologies[1:]

        if self_loop == True:
            identity = np.zeros(topologies.shape)
            for data_num in range(topologies.shape[0]):
                for rowcol in range(topologies.shape[1]):
                    if np.sum(topologies[data_num, rowcol]) != 0:
                        identity[data_num, rowcol, rowcol] = 1
            topologies = topologies + identity

        return topologies

    # get names data

    def get_names(self) -> np.ndarray:
        """
        Output stored name.

        Args:
            None
        Returns:
            names: an array of read files name. type <numpy.ndarray>
        """
        return self.names[1:]

    # get charges data

    def get_charges(self, matrix: bool = False) -> np.ndarray:
        """
        Output stored charges data

        Args:
            Matrix: if true, charge info will be presented in matrix. type <bool>
        Returns:
            charges: an array contains charges distribution. type <numpy.ndarray>
        """

        chagre_info = self.charges[1:]

        if matrix == True:
            charge_matrix = np.zeros(
                (1, self.max_atom, self.max_atom), dtype=float)
            data_num = self.get_data_shape().max()
            print('Gdata: reshaping charge data...')
            for i in tqdm(range(data_num)):
                charge_matrix_temp = np.zeros(
                    (self.max_atom, self.max_atom), dtype=float)
                for j in range(self.max_atom):
                    charge_matrix_temp[j][j] = chagre_info[i][j]
                charge_matrix = np.append(
                    charge_matrix, [charge_matrix_temp], axis=0)

            return charge_matrix[1:]

        else:
            return chagre_info

    # get structure data

    def get_structures(self) -> np.ndarray:
        """
        Output stored structures data

        Args:
            None
        Returns:
            structures: an array contains structural coordinates. type <numpy.ndarray>
        """
        return self.structures[1:]

    # load data from npy

    def load(self,
             structure_name: str = None,
             charge_name: str = None,
             name_name: str = None,
             topology_name: str = None,
             dipole_name: str = None):
        """
        Load saved structure, charge and name data from .npy file.
        Warning: this will erase all exist data in class

        Args:
            structure_name: path and name of saved structure data. type <str>
            charge_name: path and name of saved charge data. type <str>
            name_name: path and name of saved name data. type <str>
            topology_name: path and name of saved topology data. type <str>
            dipole_name: path and name of saved dipole moment data. type <str>
        Returns:
            None
        """
        structures = None
        charges = None
        names = None
        topologies = None
        dipoles = None

        if structure_name != None:
            structures = np.load(structure_name)
            max_atom = structures.shape[1]
            data_num = structures.shape[0]
        if charge_name != None:
            charges = np.load(charge_name)
            max_atom = charges.shape[1]
            data_num = charges.shape[0]
        if name_name != None:
            names = np.load(name_name)
            data_num = names.shape[0]
        if topology_name != None:
            topologies = np.load(topology_name)
            max_atom = topologies.shape[1]
            data_num = topologies.shape[0]
        if dipole_name != None:
            dipoles = np.load(dipole_name)
            data_num = dipoles.shape[0]

        # data check
        if self.__data_check(structures, charges, names, topologies, dipoles) == True:
            # get max_atom
            self.max_atom = max_atom
            # reinitialise value
            self.structures = np.zeros((1, self.max_atom, 4), dtype=float)
            self.charges = np.zeros((1, self.max_atom), dtype=float)
            self.topologies = np.zeros(
                (1, self.max_atom, self.max_atom), dtype=int)
            self.names = np.zeros(1, dtype=str)
            self.dipoles = np.zeros((1, 3), dtype=float)
            # mount data to class
            verbose_str = ''
            if structure_name != None:
                self.structures = np.append(
                    self.structures, structures, axis=0)
                verbose_str = verbose_str + ', ' + structure_name
            if charge_name != None:
                self.charges = np.append(self.charges, charges, axis=0)
                verbose_str = verbose_str + ', ' + charge_name
            if name_name != None:
                self.names = np.append(self.names, names)
                verbose_str = verbose_str + ', ' + name_name
            if topology_name != None:
                self.topologies = np.append(
                    self.topologies, topologies, axis=0)
                verbose_str = verbose_str + ', ' + topology_name
            if dipole_name != None:
                self.dipoles = np.append(self.delete_dipole, dipoles, axis=0)
                verbose_str = verbose_str + ', ' + dipole_name

            verbose_str = verbose_str[2:]

            print('Gdata:', data_num, 'data loaded and validated.',
                      'Maximum allowed atom changed to', self.max_atom)

    # save data as npy

    def save(self,
             structure_name: str = None,
             charge_name: str = None,
             name_name: str = None,
             topology_name: str = None,
             dipole_name: str = None):
        """
        Save data as numpy .npy data

        Args:
            structure_name: path and name of structure data. type <str>
            charge_name: path and name of charge data. type <str>
            name_name: path and name of name data. type <str>
            topology_name: path and name of topology data. type <str>
            dipole_name: path and name of dipole moment data. type <str>
        Returns:
            None
        """
        verbose_str = ''
        if structure_name != None:
            structures = self.structures[1:]
            np.save(structure_name, structures)
            verbose_str = verbose_str + ', ' + structure_name
        if charge_name != None:
            charges = self.charges[1:]
            np.save(charge_name, charges)
            verbose_str = verbose_str + ', ' + charge_name
        if name_name != None:
            names = self.names[1:]
            np.save(name_name, names)
            verbose_str = verbose_str + ', ' + name_name
        if topology_name != None:
            topologies = self.topologies[1:]
            np.save(topology_name, topologies)
            verbose_str = verbose_str + ', ' + topology_name
        if dipole_name != None:
            dipoles = self.dipoles[1:]
            np.save(dipole_name, dipoles)
            verbose_str = verbose_str + ', ' + dipole_name

        verbose_str = verbose_str[2:]
        print('Gdata: Data saved as: ', verbose_str)

    # read files in dir

    def read_log_dir(self, dir_name: str, validation=True):
        """
        Read Gaussian log files from directory and store in self.structure, 
        self.charges, self.name and self.dipoles

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
                    print('Gdata: file', file_name,
                            'did not pass validation! Skipped!')
                    fail_num = fail_num + 1
                    file.close()
                    continue

            try:
                structure, charge, dipole = self.__read_log(file)
            except:
                print('Gdata: Fail to read file', full_name, 'Skipped!')
                fail_num = fail_num + 1
                continue
            real_name = self.__find_real_name(file_name)
            # append this structure to list
            self.structures = np.append(
                self.structures, [structure], axis=0)
            # append this charge to list
            self.charges = np.append(self.charges, [charge], axis=0)
            # append this name to list
            self.names = np.append(self.names, real_name)
            # append this dipole to list
            self.dipoles = np.append(self.dipoles, [dipole], axis=0)
            file.close()

        print('Gdata:', file_num - fail_num,
                'log files read successfully!', fail_num, 'failed')

    # read single file

    def read_log_file(self, file_name: str, validation=True):
        """
        Read single Gaussian log file store in self.structure, 
        self.charges, self.name and self.dipoles

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
                print('Gdata: file', file_name,
                      'did not pass validation! Aborted!')
                file.close()
                raise

        # read data from file
        structure, charge, dipole = self.__read_log(file)
        real_name = self.__find_real_name(file_name)
        # append this structure to list
        self.structures = np.append(self.structures, [structure], axis=0)
        # append this charge to list
        self.charges = np.append(self.charges, [charge], axis=0)
        # append this name to list
        self.names = np.append(self.names, real_name)
        # append this dipole moment to list
        self.dipoles = np.append(self.dipoles, [dipole], axis=0)
        file.close()


        print('Gdata: log file successfully read:', file_name)


def gdata(max_atom=100,
          charge_type = 'Mulliken',
          structures: np.ndarray = None,
          charges: np.ndarray = None,
          names: np.ndarray = None,
          topologies: np.ndarray = None,
          dipoles: np.ndarray = None,
        ) -> Gdata:
    """
    Build Gdata dataframe

    Args:
        max_atom: maximum allowed atom in each structure. type <int>
        structures: array of structural coordinates. type <numpy.ndarray>
        charges: array of charges distribution. type <numpy.ndarray>
        names: array of names of structures. type <numpy.ndarray>
        topologies: array of topological information. type <numpy.ndarray>
        dipoles: array of dipole moments. type <numpy.ndarray>
    Returns:
        gd: generated Gdata dataframe
    """
    gd = Gdata(max_atom=max_atom, charge_type=charge_type)

    # data store
    if type(structures) == np.ndarray:
        gd.structures = np.append(gd.structures, structures, axis=0)
    if type(charges) == np.ndarray:
        gd.charges = np.append(gd.charges, charges, axis=0)
    if type(names) == np.ndarray:
        gd.names = np.append(gd.names, names)
    if type(topologies) == np.ndarray:
        gd.topologies = np.append(gd.topologies, topologies, axis=0)
    if type(dipoles) == np.ndarray:
        gd.dipoles = np.append(gd.dipoles, dipoles, axis=0)

    if structures == charges == names == topologies == None:
        return gd
    # data check
    gd.self_check()

    return gd


def merge(Gdata_a: Gdata, Gdata_b: Gdata) -> Gdata:
    """
    Merge two Gdata class by joining data according to name

    Args:
        Gdata_a: Gdata type to be joined. type <Gdata>
        Gdata_b: Gdata type to join. type <Gdata>
    Returns:
        Gdata_final: merged Gdata class. type <Gdata>
    """

    # copy class
    Gdata1 = Gdata_a
    Gdata2 = Gdata_b

    # data check
    if Gdata1.self_check() == False or Gdata2.self_check() == False:
        print('Gdata: merge failed! Data check not pass.')
        raise

    # initialise class

    Gdata1.minimise()
    Gdata2.minimise()

    new_max_atom = max(Gdata1.max_atom, Gdata2.max_atom)

    Gdata1.change_max_atom(new_max_atom)
    Gdata2.change_max_atom(new_max_atom)

    Gdata1.pad_zeros()
    Gdata2.pad_zeros()

    Gdata_final = gdata(new_max_atom)

    # get data from Gdata1 and Gdata2
    structure1 = Gdata1.get_structures()
    charge1 = Gdata1.get_charges()
    name1 = Gdata1.get_names()
    topology1 = Gdata1.get_topologies()
    dipole1 = Gdata1.get_dipole(style='xyz')
    data_num_1 = Gdata1.get_data_shape().max()

    structure2 = Gdata2.get_structures()
    charge2 = Gdata2.get_charges()
    name2 = Gdata2.get_names()
    topology2 = Gdata2.get_topologies()
    dipole2 = Gdata2.get_dipole(style='xyz')
    data_num_2 = Gdata2.get_data_shape().max()

    # compared name to merge
    print('Gdata: start merging data...')
    for loc1 in tqdm(range(data_num_1)):
        name_temp = name1[loc1]
        loc2_array = np.where(name2 == name_temp)

        # finding same name in Gdata2
        if loc2_array[0].ndim != 1:
            print('Gdata: Dimension error!')
            raise
        if loc2_array[0].shape[0] > 1:
            print('Gdata: more than one name matched! Name:', name_temp)
            raise
        elif loc2_array[0].shape[0] == 1:

            loc2 = loc2_array[0][0]

            # structure
            if np.array_equal(structure1[loc1], structure2[loc2]) == True:
                strcture_temp = structure1[loc1]
            elif np.count_nonzero(structure1[loc1]) * np.count_nonzero(structure2[loc2]) == 0:
                strcture_temp = structure1[loc1] + structure2[loc2]
            else:
                print('Gdata: Conflict found between two sets of data.')
                print('Gdata: Structure data conflict in %s' % (name1[loc1]))
                print('Gdata: Data in first data set: \n', structure1[loc1], sep='')
                print('Gdata: Data in Second data set: \n', structure2[loc2], sep='')
                raise

            # charge
            if np.array_equal(charge1[loc1], charge2[loc2]) == True:
                charge_temp = charge1[loc1]
            elif np.count_nonzero(charge1[loc1]) * np.count_nonzero(charge2[loc2]) == 0:
                charge_temp = charge1[loc1] + charge2[loc2]
            else:
                print('Gdata: Conflict found between two sets of data.')
                print('Gdata: Charge data conflict in %s' % (name1[loc1]))
                print('Gdata: Data in first data set: \n', charge1[loc1], sep='')
                print('Gdata: Data in Second data set: \n', charge2[loc2], sep='')
                raise

            # topology
            if np.array_equal(topology1[loc1], topology2[loc2]) == True:
                topology_temp = topology1[loc1]
            elif np.count_nonzero(topology1[loc1]) * np.count_nonzero(topology2[loc2]) == 0:
                topology_temp = topology1[loc1] + topology2[loc2]
            else:
                print('Gdata: Conflict found between two sets of data.')
                print('Gdata: Topology data conflict in %s' % (name1[loc1]))
                print('Gdata: Data in first data set: \n', topology1[loc1], sep='')
                print('Gdata: Data in Second data set: \n', topology2[loc2], sep='')
                raise
            

            # dipole
            if np.array_equal(dipole1[loc1], dipole1[loc2]) == True:
                dipole_temp = dipole1[loc1]
            elif np.count_nonzero(dipole1[loc1]) * np.count_nonzero(dipole2[loc2]) == 0:
                dipole_temp = dipole1[loc1] + dipole2[loc2]
            else:
                print('Gdata: Conflict found between two sets of data.')
                print('Gdata: Dipole moment data conflict in %s' % (name1[loc1]))
                print('Gdata: Data in first data set: \n', dipole1[loc1], sep='')
                print('Gdata: Data in Second data set: \n', dipole2[loc2], sep='')
                raise

            # remove item in data2
            structure2 = np.delete(structure2, loc2, axis=0)
            charge2 = np.delete(charge2, loc2, axis=0)
            name2 = np.delete(name2, loc2, axis=0)
            topology2 = np.delete(topology2, loc2, axis=0)
            dipole2 = np.delete(dipole2, loc2, axis=0)
            # add in gdata_final
            Gdata_final.add_data(strcture_temp, charge_temp,
                                 name_temp, topology_temp, dipole_temp)

    # add rest data to gdata final
    Gdata_final.add_data(structure2, charge2, name2, topology2, dipole2)

    return Gdata_final
