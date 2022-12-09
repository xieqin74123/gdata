# Introduction
Gaussian data processing tool. Designed for large number of Gaussian .log files read/convert/manage operations. Used as machine learning data pre-processing tool.

# Dependencies
- numpy
- tqdm
- bidict

# Usage
'gdata_main.py' is the terminal based GUI for basic usage. This should be located on the same folder of 'gdata.py' and can be run simply from terminal.
'gdata.py' is the source that contains all Gdata class definitions and methods, and relevant functions. It can be used directly by adding `import gdata as gdata` in your python script for advance usage.

# API Reference
## Function element_dic(sym)
    Two way dictionary for atomic number and symbol

    Args:
        sym: atomic number or symbol. type <int> or <str>
    Returns:
        atomic_number: if input is a symbol, return atomic number. type <int>
        atomic_number.inverse: if input is a number, return atomic symbol. type <str>
       
## Function atom_mass_dict(sym)
    One way dictionary from atom type to atomic mass

    Args:
        sym: atomic number or symbol. type <int>, <float> or <str>
    Returns:
        atomic weight: type <float>

## Function vec_normalise(vec)
    Normalise a vector

    Args:
        vec: vector to be nomalised. type <numpy.ndarray>
    Returns:
        n_vec: normalised vector. type <numpy.ndarray>
        
## Function gdata(max_atom=100, charge_type='Mulliken', structures=None, charges=None, names=None, topologies=None, dipoles=None)
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
        
## Function merge(Gdata_a, Gdata_b)
    Merge two Gdata class by joining data according to name

    Args:
        Gdata_a: Gdata type to be joined. type <Gdata>
        Gdata_b: Gdata type to join. type <Gdata>
    Returns:
        Gdata_final: merged Gdata class. type <Gdata>
        
## Class Gdata()
### Initialisation __init__(self, charge_type='Mulliken', max_atom=100)
        Initialise Gdata class.

        Args:
            charge_type: define the charge type used in this Gdata class. type <str>
                Alternative: Hirshfeld, Mulliken
            max_atom: maximum allowed atoms in single molecule. type <int>
            
### Private Function __data_check(self, structures=None, charges=None, names=None, topologies=None, dipoles=None)
        Check input data whether they match each other and the requirement

        Args:
            structures: array of structure data. type <np.ndarray>
            charges: array of charge data. type <np.ndarray>
            names: array of name data. type <np.ndarray>
            topologies: array of topology data. type <np.ndarray>
            dipoles: array of dipole moment data. type <np.ndarray>
        Returns:
            check_result: if data pass check, return True. type <bool>

### Private Function __read_zmat(self, file)
        read in .com file generate by Gaussian newzmat

        Args:
            file: opened file to read in. type <TextIOWrapper>
        Returns:
            coordinate: xyz coordinate of molecule. type <numpy.ndarray>
            topologies: topological information of bonding. type <numpy.ndarray>

### Private Function __write_xyz(self, file, coordinate)
        write xyz coordinate to file and return actual number of atom in molecule

        Args:
            file: file to write. type <_io.TextIOWrapper>
            coordinate: coordinate of structure in shape [max_atom, 4]. type <numpy.ndarray>
        Returns:
            atom_num: actual number of atom in this structure. type <int>
            
### Private Function __read_xyz(self, file, header=True)
        Read coordinate from .xyz file

        Args:
            file: opened xyz file. type <_io.TextIOWrapper>
            hearder: indicate whether the reading xyz file contain header or not. type <bool>
        Returns:
            coordinate: structural coordinate from xyz file. type <numpy.ndarray>

### Private Function __find_real_name(self, file_name)
        Get file name without path and extension

        Args:
            file_name: name of file including path and extension. type <str>
        Returns:
            pure_file_name: file name without path and extension. type <str>

### Private Function __val__log(self, file)
        Validate Gaussian log file if it was exit normally

        Args:
            file: opened file for validation. type <_io.TextIOWrapper>
        Returns:
            validation_result: True for passed. type <bool>

### Private Function __read_log(self, file)
        Read coordinate and charge from Gaussian output log file

        Args:
            file: opened log file to read. type <_io.TextIOWrapper>
        Returns:
            structure: coordinate strcture in log file. type <numpy.ndarray>
            charge: charge information in log file. type <numpy.ndarray>
            dipole_moment: dipole moment in log file. type <float>

### Public Function convert_to_mi_coordinate(self)
        Convert input coordinates to moment of inertial tensor eigen vector based unified coordinates. 
        Warning: This will overwrite existing structure data
        
        Args:
            None
        Returns:
            mi_structure: type <numpy.ndarray>
            
### Public Function get_moment_of_inertia_tensor(self)
        Ouput moment of inertial tensor of all these molecules
        
        Args:
            None
        Returns:
            inert_tensor: type <numpy.ndarray>
            
### Public Function get_atom_weight(self)
        Get atom weight info

        Args:
            None
        Return:
            atom_weight: type <numpy.ndarray>
            
### Public Function get_mass_centre(self)
        Get coordinate of mass centre

        Args:
            None
        Returns:
            mass centre: type <numpy.ndarray>
            
### Public Function add_data(self, structure=None, charge=None, name=None, topology=None, dipole=None)
        Add data to class manully

        Args:
            structure: array of structural coordinates. type <numpy.ndarray>
            charge: array of charges distribution. type <numpy.ndarray>
            name: array of names of structures. type <numpy.ndarray>
            topology: array of topological information. type <numpy.ndarray>
            dipole: array of dipole moment data. type <numpy.ndarray>
        Returns:
            None
            
### Public Function pad_zeros(self)
        Pad zeros to thoes blank without data to fit in shape

        Args:
            None
        Returns:
            None
            
### Public Function delete_dipole(self)
        Delete all dipole moment info in class

        Args:
            None
        Return:
            None
            
### Public Function delete_topologies(self)
        Delete all topology data in calss

        Args:
            None
        Returns:
            None
            
### Public Function delete_names(self)
        Delete all name data in calss

        Args:
            None
        Returns:
            None
            
### Public Function delete_charges(self)
        Delete all charge data in class

        Args:
            None
        Returns:
            None
            
### Public Function delete_structures(self)
        Delete all structure data in class

        Args:
            None
        Returns:
            None
            
### Public Function change_max_atom(self, new_max_atom)
        Change max_atom number in class

        Args:
            new_max_atom: desired maximum allowed atom in molecule. type <int>
        Returns:
            None
            
### Public Function self_check(self)
        Check stored data whether valid or not

        Args:
            None
        Return:
            check_result: if True, data has passed the check. type <bool>
            
### Public Function read_zmat_dir(self, dir_name)
        Read all zmat files from directory

        Args:
            dir_name: path and name of the directory. type <str>
        Returns:
            None
            
### Public Function read_zmat_file(self, file_name)
        Read single newzmat .com file

        Args:
            file_name: path and name of file. type <str>
        Return:
            None
            
### Public Function minimise(self)
        Reduce maximum allowed atom to minimise the shape of array

        Args:
            None
        Return:
            None
            
### Public Function convert_to_xyz(self, directory, header=True)
        Convert stored structure data to xyz format

        Args:
            directory: directory of output files. type <str>
            header: if True, print .xyz header. type <bool>
        Returns:
            None
            
### Public Function get_data_shape(self)
        Get number of data stored in class, for structures, charges, names, topologies and dipole moments

        Args:
            None
        Return:
            shape_array: an array of shape in order of [structures, charges, names, topologies, dipole moments]. type <numpy.ndarray>
            
### Public Function read_xyz_dir(self, dir_name, header=True)
        Read all xyz files in the directory. Store in self.structures and self.names

        Args:
            dir_name: path/name of directory. type <str>
        Returns:
            None
            
### Public Function read_xyz_file(self, file_name, header=True)
        Read single xyz file and store in self.structure and self.name

        Args:
            file_name: the name of file to read. type <str>
        Returns:
            None
            
### Public Function get_dipole(self, style='norm')
        Output dipole moment

        Args:
            style: select output style. <str>
                Alternative:
                    norm: output the norm of dipole moment
                    xyz: output dipole in x, y and z directions
        Return:
            dipole: dipole moment information. type <numpy.ndarray>
            
### Public Function get_degree(self)
        Output degree matrix for GCN

        Args:
            None
        Returns:
            degree: degree matrix for GCN. type <numpy.ndarray>
            
### Public Function get_adjacency(self, self_loop=False)
        Output adjacency for GCN

        Args:
            self_loop: add self-bond to adjacency matrix. type <bool>
        Returns:
            adjacency: adjacency matrix for GCN. type <numpy.ndarray>
            
### Public Function get_atom_info(self, format='array')
        Output atom type in order

        Args:
            format: output format. type <str>
                Alternative:
                    array: default value. output atomic number in array
                    matrix: output atomic number in matrix
        Returns:
            atom_info: atomic number in array or matrix. type <numpy.ndarray>
            
### Public Function get_topologies(self, self_loop=False)
        Output stored topological data

        Args:
            self_loop: add self loop to topology matrix. type <bool>
        Returns:
            topologies: and matrix of topologies info. type <numpy.ndarray>
            
### Public Function get_names(self)
        Output stored name.

        Args:
            None
        Returns:
            names: an array of read files name. type <numpy.ndarray>
            
### Public Function get_charges(self, matrix=False)
        Output stored charges data

        Args:
            style: output style
                Alternative:
                    array: output charge in array
                    matrix: output charge in diagonal element of a matrix
        Returns:
            charges: an array contains charges distribution. type <numpy.ndarray>
            
### Public Function get_structures(self, coor_only=False)
        Output stored structures data

        Args:
            coor_only: output coordinates info only. type <bool>
        Returns:
            structures: an array contains structural coordinates. type <numpy.ndarray>
            
### Public Function load(self, structure_name=None, charge_name=None, name_name=None, topology_name=None, dipole_name=None)
        Load saved structure, charge and name data from .npy file.
        Warning: this will erase all exist data in class. Create a new class and use merge to keep existing data.

        Args:
            structure_name: path and name of saved structure data. type <str>
            charge_name: path and name of saved charge data. type <str>
            name_name: path and name of saved name data. type <str>
            topology_name: path and name of saved topology data. type <str>
            dipole_name: path and name of saved dipole moment data. type <str>
        Returns:
            None
         
### Public Function save(self, structure_name=None, charge_name=None, name_name=None, topology_name=None, dipole_name=None)
        Save data as numpy .npy files

        Args:
            structure_name: path and name of structure data. type <str>
            charge_name: path and name of charge data. type <str>
            name_name: path and name of name data. type <str>
            topology_name: path and name of topology data. type <str>
            dipole_name: path and name of dipole moment data. type <str>
        Returns:
            None
            
            
### Public Function read_log_dir(self, dir_name, validation=True)
        Read Gaussian log files from directory and store in self.structure, 
        self.charges, self.name and self.dipoles

        Args:
            dir_name: path and name of directory. type <str>
            validation: enable validation for log, this will check whether the Gaussian calculation exit normally in this log file. type <bool>
        Returns:
            None
            
### Public Function read_log_file(self, file_name, validation=True)
        Read single Gaussian log file store in self.structure, 
        self.charges, self.name and self.dipoles

        Args:
            file_name: path and name of Gaussian log file. type <str>
            validation: enable validation for log, this will check whether the Gaussian calculation exit normally in this log file. type <bool>
        Returns:
            None
