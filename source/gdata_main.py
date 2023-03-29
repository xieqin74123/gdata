import gdata
import os
import platform
import numpy as np

version = 'master'

def clear ():
    global clear_arg
    os.system(clear_arg)

def input_command (cmd:str='User Command:', numeric_check=True):
    ucommand = input('\033[7m %s \033[0m ' % (cmd))
    if numeric_check == True:
        try: 
            ucommand = int(ucommand)
            return ucommand
        except Exception as e:
            return None
    return ucommand


def print_welcome ():
    global pltf
    global version

    print('Platform identified as %s' % (pltf))
    print('Welcome to Gdata TUI, version: %s' % (version))

def main_menu ():
    global gdata_list
    global gdata_count
    global gdata_name
    global first_time

    clear()

    if first_time == 1:
        first_time = 0
        print_welcome()

    # title
    print('--- MAIN MENU ---')

    list_num = -1
    print('OPERATIONS:')
    print('%d \t - \t Refresh' % (list_num))
    list_num += 1
    print('%d \t - \t Exit' % (list_num))
    list_num += 1
    print('%d \t - \t New Gdata Class' % (list_num))    # append gd function
    list_num += 1
    print('%d \t - \t Merge Gdata' % (list_num))
    list_num += 1
    gd_list_start = list_num
    print('DATA:')  # list gd data
    for i in range(len(gdata_list)):
        print('%d \t - \t Gdata %s' % (list_num, gdata_name[i]))
        list_num += 1
    
    return gd_list_start, list_num

def gdata_menu_title (selected_data: int):
    global gdata_name
    global gdata_list
    clear()
    print('--- Gdata Menu %s ---' % (gdata_name[selected_data]))

    # functions
    print('OPERATIONS:')
    print('-1 \t - \t Refresh')
    print('0 \t - \t Back')
    print('1 \t - \t Delete Menu')
    print('2 \t - \t Change Gdata Settings')
    print('3 \t - \t Rename')
    print('4 \t - \t Read Gaussian log')
    print('5 \t - \t Read Gaussian zmat')
    print('6 \t - \t Read mol File')
    print('7 \t - \t Read mol2 File')
    print('8 \t - \t Save Data as .npy')
    print('9 \t - \t Laod .npy Data')
    print('10 \t - \t Manage .xyz Structure Data')
    print('11 \t - \t Convert to MI Based Coordinates')

    # information
    print('INFORMATION:')
    print('Maximum allowed atom: \t %d' % (gdata_list[selected_data].max_atom))
    print('Number of data: \t %d' % (gdata_list[selected_data].get_data_shape()[0]))
    print('Charge type: \t \t %s' % (gdata_list[selected_data].charge_type))
    mi_coor = 'N'
    if gdata_list[selected_data].mi_coor is True:
        mi_coor = 'Y'
    print('Mi coordinates: \t %c' % (mi_coor))

    print('')
    structures_exist, charges_exist, names_exist, topologies_exist, dipoles_exist = data_exist_check(selected_data)
    
    print('DATA EXISTANCE:')
    print('XYZ Struacture: \t %c' % (structures_exist))
    print('Charge Data: \t \t %c' % (charges_exist))
    print('Name Data: \t \t %c' % (names_exist))
    print('Topology Info: \t \t %c' % (topologies_exist))
    print('Dipole Moment: \t \t %c' % (dipoles_exist))


def gdata_menu (selected_data: int):
    global gdata_list
    global gdata_count
    global gdata_name

    gdata_menu_title(selected_data)
    while(1):
        ucommand = input_command()
        # standard operations
        if ucommand == -1:      # refresh
            gdata_menu_title(selected_data)
            continue
        elif ucommand == 0:     # back
            return 0
        elif ucommand == 1:     # delete
            if gdata_delete(selected_data) == 1:
                return 0
            gdata_menu_title(selected_data)
        elif ucommand == 2:     # minimise
            change_settings(selected_data)
            gdata_menu_title(selected_data)
        elif ucommand == 3:
            new_name = input_command('New name:', numeric_check=False)
            gdata_name[selected_data] = new_name
            gdata_menu_title(selected_data)
        elif ucommand == 4:     # read log
            read_log(selected_data)
            gdata_menu_title(selected_data)
        elif ucommand == 5:     # read zmat
            read_zmat(selected_data)
            gdata_menu_title(selected_data)
        elif ucommand == 6:     # read mol
            read_mol(selected_data)
            gdata_menu_title(selected_data)
        elif ucommand == 7:     # read mol2
            read_mol2(selected_data)
            gdata_menu_title(selected_data)
        elif ucommand == 8:     # save data
            data_save_load(selected_data, 'save')
            gdata_menu_title(selected_data)
        elif ucommand == 9:     # load data
            data_save_load(selected_data, 'load')
            gdata_menu_title(selected_data)
        elif ucommand == 10:     # manage xyz
            manage_xyz(selected_data)
            gdata_menu_title(selected_data)
        elif ucommand == 11:
            try:
                gdata_list[selected_data].convert_to_mi_coordinate()
            except Exception as e:
                print(e)
            input_command('Press Enter to Continue', numeric_check=False)
            gdata_menu_title(selected_data)
        # advanced operations
        elif ucommand == -999:
            clear()
            cat_data(selected_data)
            gdata_menu_title(selected_data)

        # error
        else:
            print('Invalid Input!')


def read_mol2 (selected_data:int):
    global gdata_list

    read_mol2_title(selected_data)

    while(1):
        ucommand = input_command()
        if ucommand == -1:  # Refresh
            read_mol2_title(selected_data)
        elif ucommand == 0: # Back
            return 0
        elif ucommand == 1: # Read Single mol2 File
            path = input_command('Path:', numeric_check=False)
            try:
                gdata_list[selected_data].read_mol2_file(path)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
                clear()
        elif ucommand == 2: # Read mol2 files from Folder
            path = input_command('Path:', numeric_check=False)
            try:
                gdata_list[selected_data].read_mol2_dir(path)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
                clear()
        # error
        else:
            print('Invalid Input!')

def read_mol2_title (selected_data):
    global gdata_name

    clear()
    print('Read mol2 file to %s' % (gdata_name[selected_data]))
    print('OPERATIONS:')
    print('-1 \t - \t Refresh')
    print('0 \t - \t Back')
    print('1 \t - \t Read Single mol2 File')
    print('2 \t - \t Read mol2 Files from Floder')

def read_mol (selected_data:int):
    global gdata_list

    read_mol_title(selected_data)

    while(1):
        ucommand = input_command()
        if ucommand == -1:  # Refresh
            read_mol_title(selected_data)
        elif ucommand == 0: # Back
            return 0
        elif ucommand == 1: # Read Single mol File
            path = input_command('Path:', numeric_check=False)
            try:
                gdata_list[selected_data].read_mol_file(path)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
                clear()
        elif ucommand == 2: # Read mol files from Folder
            path = input_command('Path:', numeric_check=False)
            try:
                gdata_list[selected_data].read_mol_dir(path)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
                clear()
        # error
        else:
            print('Invalid Input!')

def read_mol_title (selected_data:int):
    global gdata_name

    clear()
    print('Read mol file to %s' % (gdata_name[selected_data]))
    print('OPERATIONS:')
    print('-1 \t - \t Refresh')
    print('0 \t - \t Back')
    print('1 \t - \t Read Single mol File')
    print('2 \t - \t Read mol Files from Floder')

def gdata_delete (selected_data:int):
    global gdata_list
    global gdata_name
    gdata_delete_title(selected_data)

    while(1):
        ucommand = input_command()

        if ucommand == -1:  # Refresh
            gdata_delete_title(selected_data)
        elif ucommand == 0:   # Back
            return 0
        elif ucommand == 1: # Delete Class
            del_confirm = input_command('Type \'yes\' to confirm:', numeric_check=False)
            if del_confirm == 'yes':
                gdata_list.pop(selected_data)
                gdata_name.pop(selected_data)
                return 1
            else:
                print('Aborted')
        elif ucommand == 2: #Delete Structure Data
            del_confirm = input_command('Type \'yes\' to confirm:', numeric_check=False)
            if del_confirm == 'yes':
                gdata_list[selected_data].delete_structures()
                return 0
            else:
                print('Aborted')
        elif ucommand == 3: #Delete Charge Data
            del_confirm = input_command('Type \'yes\' to confirm:', numeric_check=False)
            if del_confirm == 'yes':
                gdata_list[selected_data].delete_charges()
                return 0
            else:
                print('Aborted')
        elif ucommand == 4: #Delete topology Data
            del_confirm = input_command('Type \'yes\' to confirm:', numeric_check=False)
            if del_confirm == 'yes':
                gdata_list[selected_data].delete_topologies()
                return 0
            else:
                print('Aborted')
        elif ucommand == 5: #Delete Dipole Data
            del_confirm = input_command('Type \'yes\' to confirm:', numeric_check=False)
            if del_confirm == 'yes':
                gdata_list[selected_data].delete_dipole()
                return 0
            else:
                print('Aborted')
        # error
        else:
            print('Invalid Input!')

def gdata_delete_title (selected_data:int):
    global gdata_name
    clear()
    print('--- Delete Data in %s ---' % (gdata_name[selected_data]))
    print('OPERATIONS:')
    print('-1 \t - \t Refresh')
    print('0 \t - \t Back')
    print('1 \t - \t Delete this Class')
    print('2 \t - \t Delete Structure Data')
    print('3 \t - \t Delete Charge Data')
    print('4 \t - \t Delete Topology Data')
    print('5 \t - \t Delete Dipole Moment Data')

    structures_exist, charges_exist, names_exist, topologies_exist, dipoles_exist = data_exist_check(selected_data)
    print('DATA EXISTANCE:')
    print('XYZ Struacture: \t %c' % (structures_exist))
    print('Charge Data: \t \t %c' % (charges_exist))
    print('Name Data: \t \t %c' % (names_exist))
    print('Topology Info: \t \t %c' % (topologies_exist))
    print('Dipole Moment: \t \t %c' % (dipoles_exist))

def data_exist_check (selected_data:int):
    global gdata_list

    # Data Exist Check
    structures_exist = 'N'
    charges_exist = 'N'
    names_exist = 'N'
    topologies_exist = 'N'
    dipoles_exist = 'N'

    if np.count_nonzero(gdata_list[selected_data].structures) > 0:
        structures_exist = 'Y'
    if np.count_nonzero(gdata_list[selected_data].charges) > 0:
        charges_exist = 'Y'
    if gdata_list[selected_data].names.shape[0] > 1:
        names_exist = 'Y'
    if np.count_nonzero(gdata_list[selected_data].topologies) > 0:
        topologies_exist = 'Y'
    if np.count_nonzero(gdata_list[selected_data].dipoles) > 0:
        dipoles_exist = 'Y'

    return structures_exist, charges_exist, names_exist, topologies_exist, dipoles_exist

def manage_xyz (selected_data:int):
    global gdata_list
    global gdata_name
    header = True
    manage_xyz_title(selected_data, header)
    while(1):
        ucommand = input_command()
        if ucommand == -1:      # Refresh
            manage_xyz_title(selected_data, header)
            continue
        elif ucommand == 0:     # Back
            return 0
        elif ucommand == 1:     # read single xyz
            path = input_command('Path:', numeric_check=False)
            try: 
                gdata_list[selected_data].read_xyz_file(path, header)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
        elif ucommand == 2:     # read xyz from dir
            path = input_command('Path:', numeric_check=False)
            try:
                gdata_list[selected_data].read_xyz_dir(path, header)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
        elif ucommand == 3:     # save xyz to dir
            path = input_command('Path:', numeric_check=False)
            try: 
                gdata_list[selected_data].convert_to_xyz(path, header)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
        elif ucommand == 4:     # change header setting
            if header is True:
                header = False
            elif header is False:
                header = True
            manage_xyz_title(selected_data, header)
        else:
            print('Invalid Input!')

def manage_xyz_title (selected_data:int, header:bool):
    global gdata_list
    global gdata_name
    clear()
    print('--- Manage XYZ in %s ---' % (gdata_name[selected_data]))
    print('OPERATIONS:')
    print('-1 \t - \t Refresh')
    print('0 \t - \t Back')
    print('1 \t - \t Read Single .xyz File')
    print('2 \t - \t Read .xyz Files from Folder')
    print('3 \t - \t Save Structure Data as .xyz file')
    print('4 \t - \t Change Header Setting')
    print('INFORMATION:')
    print('Warning: xyz format only contains structure info')
    print('Header Setting: \t %s' % (header))


def cat_data (selected_data:int):
    global gdata_list
    global gdata_name

    print('--- DEV MODE CAT (%s) ---' % (gdata_name[selected_data]))
    print('Structure:')
    print(gdata_list[selected_data].structures[1:])
    print('Charge:')
    print(gdata_list[selected_data].charges[1:])
    print('name:')
    print(gdata_list[selected_data].names[1:])
    print('topology:')
    print(gdata_list[selected_data].topologies[1:])
    print('dipole:')
    print(gdata_list[selected_data].dipoles[1:])
    input_command('Press Enter to Continue', numeric_check=False)

def new_gdata ():
    global gdata_list
    global gdata_count
    global gdata_name

    gdata_list.append(gdata.gdata())
    new_gd_name = 'gd_' + str(gdata_count)
    gdata_name.append(new_gd_name)
    gdata_count += 1

def read_log_title (selected_data: int, validation_check):
    global gdata_name
    clear()
    print('Read Gaussian Log to %s:' % (gdata_name[selected_data]))
    print('OPERATIONS:')
    print('-1 \t - \t Refresh')
    print('0 \t - \t Back')
    print('1 \t - \t From Single File')
    print('2 \t - \t From Directory')
    print('3 \t - \t Change Validation Setting')
    print('INFORMATION:')
    print('Validation Check: \t %s' % (validation_check))

def read_log (selected_data:int):
    global gdata_list
    validation_check = True
    read_log_title(selected_data, validation_check)
    while(1):
        ucommand = input_command()

        if ucommand == -1:  # refresh
            read_log_title(selected_data, validation_check)
            continue
        elif ucommand == 0: # back
            return 0
        elif ucommand == 1: # read from single file
            path = input_command('File Directory:', numeric_check=False)
            try:
                gdata_list[selected_data].read_log_file(path, validation=validation_check)
                input_command('Press Enter to Continue', numeric_check=False)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
                return -1
        elif ucommand == 2: # read from directory
            path = input_command('Path:', numeric_check=False)
            try:
                gdata_list[selected_data].read_log_dir(path, validation=validation_check)
                input_command('Press Enter to Continue', numeric_check=False)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
                return -1
        elif ucommand == 3: # change validation check setting
            if validation_check == True:
                validation_check = False
            elif validation_check == False:
                validation_check = True
            read_log_title(selected_data, validation_check)
        else:
            print('Invalid Input!')

def read_zmat_title (selected_data:int):
    global gdata_name
    clear()
    print('Read Gaussian Zmat to %s:' % (gdata_name[selected_data]))
    print('OPERATIONS:')
    print('-1 \t - \t Refresh')
    print('0 \t - \t Back')
    print('1 \t - \t From Single File')
    print('2 \t - \t From Directory')

def read_zmat (selected_data:int):
    global gdata_list
    read_zmat_title(selected_data)
    while(1):
        ucommand = input_command()
        if ucommand == -1:  # refresh
            read_zmat_title(selected_data)
        elif ucommand == 0: # back
            return 0
        elif ucommand == 1: # read from single file
            path = input_command('File Directory:', numeric_check=False)
            try:
                gdata_list[selected_data].read_zmat_file(path)
                input_command('Press Enter to Continue', numeric_check=False)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
                return -1
        elif ucommand == 2: # read zamt from dir
            path = input_command('Path:', numeric_check=False)
            try:
                gdata_list[selected_data].read_zmat_dir(path)
                input_command('Press Enter to Continue', numeric_check=False)
                return 0
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
                return -1
        else:
            print('Invalid Input!')

def change_settings (selected_data:int):
    global gdata_list
    change_settings_title(selected_data)
    while(1):
        ucommand = input_command()
        if ucommand == -1:      # refresh
            change_settings_title(selected_data)
        elif ucommand == 0:     # back
            return 0
        elif ucommand == 1:     # minimise
            gdata_list[selected_data].minimise()
            change_settings_title(selected_data)
        elif ucommand == 2:     # change max_atom
            new_max_atom = input_command('New Maximum Allowed Atom:')
            if new_max_atom >= 0:
                gdata_list[selected_data].change_max_atom(new_max_atom)
                change_settings_title(selected_data)
            else:
                print('Invalid Input!')
        elif ucommand == 3:     # change charge type
            clear()
            print('Select Chagre Type:')
            print('1 \t - \t Mulliken Charge')
            print('2 \t - \t Hirshfeld Charge')
            ucommand = input_command()
            if ucommand == 1:
                gdata_list[selected_data].charge_type = 'Mulliken'
            elif ucommand == 2:
                gdata_list[selected_data].charge_type = 'Hirshfeld'
            else:
                print('Invalid Input!')
                input_command('Press Enter to Continue', numeric_check=False)
            change_settings_title(selected_data)


def change_settings_title (selected_data:int):
    global gdata_name
    global gdata_list
    clear()
    print('Change Gdata %s Settings:' % (gdata_name[selected_data]))
    print('OPERATIONS:')
    print('-1 \t - \t Refresh')
    print('0 \t - \t Back')
    print('1 \t - \t Minimise Maximum Allowed Atom (Auto)')
    print('2 \t - \t Change Maximum Allowed Atom (Manually)')
    print('3 \t - \t Change Charge Type')
    print('INFORMATION:')
    print('Maximum Allowed Atom: \t %d' % (gdata_list[selected_data].max_atom))
    print('Number of Data: \t %d' % (gdata_list[selected_data].get_data_shape()[0]))
    print('Charge Type: \t \t %s' % (gdata_list[selected_data].charge_type))

def data_save_load (selected_data:int, operation:str):
    directory = input_command('Path:', numeric_check=False)
    # check '/'
    name_len = len(directory)
    if name_len == 0:
        print('Invalid Input!')
        input_command('Press Enter to Continue', numeric_check=False)
        return -1
    if directory[name_len-1] != '/':
        directory = directory + '/'
    # check dir exist
    try:
        os.listdir(directory)
    except:
        os.mkdir(directory)
    # combine name
    structure_name = directory + 'structure.npy'
    charge_name = directory + 'charge.npy'
    name_name = directory + 'name.npy'
    topology_name = directory + 'topology.npy'
    dipole_name = directory + 'dipole.npy'
    config_name = directory + 'config.npy'
    if operation == 'save':
        gdata_list[selected_data].save(
            structure_name, charge_name, name_name, topology_name, dipole_name, config_name)
    elif operation == 'load':
        gdata_list[selected_data].load(
            structure_name, charge_name, name_name, topology_name, dipole_name, config_name)
    input_command('Press Enter to Continue', numeric_check=False)

def merge () :
    global gdata_list
    global gdata_name
    global gdata_count
    clear()
    data_1 = -1
    data_2 = -1
    merge_title(data_1, data_2)
    while(1):
        ucommand = input_command()
        if ucommand == -1:
            merge_title(data_1, data_2)
        elif ucommand == 0:
            return 0
        elif ucommand == 1:
            merge_gdata_list()
            data_temp = input_command('Select Gdata:') - 1
            if data_temp in range(len(gdata_name)):
                data_1 = data_temp
            else:
                print('Invalid Input!')
                input_command('Press Enter to Continue', numeric_check=False)
            merge_title(data_1, data_2)
        elif ucommand == 2:
            merge_gdata_list()
            data_temp = input_command('Select Gdata:') - 1
            if data_temp in range(len(gdata_name)):
                data_2 = data_temp
            else:
                print('Invalid Input!')
                input_command('Press Enter to Continue', numeric_check=False)
            merge_title(data_1, data_2)
        elif ucommand == 3:
            try:
                gdata_list.append(gdata.merge(gdata_list[data_1], gdata_list[data_2]))
                new_name = 'gd_' + str(gdata_count)
                gdata_count += 1
                gdata_name.append(new_name)
                print('%s and %s have been merged into %s.' % (gdata_name[data_1], gdata_name[data_2], new_name))
                input_command('Press Enter to Continue', numeric_check=False)
                merge_title(data_1, data_2)
            except Exception as e:
                print(e)
                input_command('Press Enter to Continue', numeric_check=False)
                merge_title(data_1, data_2)
        else:
            print('Invalid Input!')


def merge_title (data_1, data_2):
    global gdata_name
    clear()
    if data_1 == -1:
        data_1_name = 'None'
    else:
        data_1_name = gdata_name[data_1]
    if data_2 == -1:
        data_2_name = 'None'
    else:
        data_2_name = gdata_name[data_2]
    print('--- Merge menu ---')
    print('OPERATIONS:')
    print('-1 \t - \t Refresh')
    print('0 \t - \t Back')
    print('1 \t - \t Selected First Gdata: %s' % (data_1_name))
    print('2 \t - \t Selected Second Gdata: %s' % (data_2_name))
    print('3 \t - \t Merge Confirm')

def merge_gdata_list ():
    global gdata_name
    clear()
    print('OPERATION:')
    print('0 \t - \t Back')
    print('AVALIABLE GDATA:')
    list_num = 1
    for i in range (len(gdata_name)):
        print('%d \t - \t %s' % (list_num, gdata_name[i]))
        list_num += 1

if __name__ == '__main__':
    pltf = platform.system()
    if pltf == 'Windows':
        clear_arg = 'cls'
    elif pltf == 'Linux':
        clear_arg = 'clear'
    else:
        clear_arg = 'clear'
    
    clear()
    print('Initialising...')
    print('Platform identified as %s' % (pltf))
    print('Welcome to Gdata, version: %s' % (version))
    # initialisation
    gdata_list = []
    gdata_count = 0
    gdata_name = []
    new_gdata()
    first_time = 1
    gd_list_start, list_num = main_menu()
    # main menu input
    while(1):
        ucommand = input_command()
        if ucommand == -1:      # regresh
            gd_list_start, list_num = main_menu()
            continue
        elif ucommand == 0:     # exit
            clear()
            exit_confirm = input_command('Type \'exit\' to Exit:', numeric_check=False)
            if exit_confirm == 'exit':
                clear()
                exit()
            else:
                gd_list_start, list_num = main_menu()
                continue
        elif ucommand == 1:     # new gdata
            new_gdata()
            gd_list_start, list_num = main_menu()
        elif ucommand == 2:     # merge
            merge()
            gd_list_start, list_num = main_menu()
        elif ucommand in range(gd_list_start, list_num):
            selected_data = ucommand - 3
            gdata_menu(selected_data)
            selected_data = 0
            gd_list_start, list_num = main_menu()
        else:
            print('Invalid Input!')
