# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:48:22 2017

@author: josemiguelarrieta
"""
#Load libraries
import os 
os.chdir('Documents/SIVA')
import cv2

from utils_dagm import get_labels_defectA, get_labels_defectB, get_roi_rect
from utils_dagm import save_list_selected_images, load_list_selected_images, move_selected_images


################
#SELECT IMAGES #
################
#Aqui van los Arrays!.
class_number = 4
number_experimet = 1

"""
list_no_defect = select_images('',class_number)
list_defect_A = select_images('A',class_number)
list_defect_AB = select_images('AB',class_number)
list_defect_B = select_images('B',class_number)
"""

list_no_defect = ['1','2','3','4','5','6','7','8','9']
list_defect_A = ['9','20','21','23']
list_defect_AB = ['31','33','34','36']
list_defect_B = ['40','41','45','48']

save_list_selected_images(list_no_defect,'',class_number,number_experimet)
save_list_selected_images(list_defect_A,'A',class_number,number_experimet)
save_list_selected_images(list_defect_B,'B',class_number,number_experimet)
save_list_selected_images(list_defect_AB,'AB',class_number,number_experimet)

A = load_list_selected_images('',class_number,number_experimet)
B = load_list_selected_images('A',class_number,number_experimet)
C = load_list_selected_images('B',class_number,number_experimet)
D = load_list_selected_images('AB',class_number,number_experimet)

move_selected_images(list_no_defect,'',class_number,number_experimet)
move_selected_images(list_defect_A,'A',class_number,number_experimet)
move_selected_images(list_defect_AB,'AB',class_number,number_experimet)
move_selected_images(list_defect_B,'B',class_number,number_experimet)




num=1
path= '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiment_1_DAGM/Class'


get_labels_defectA(path,class_number,num,exp=True,defect='A',degrees=True)
get_labels_defectB(path,class_number,num,exp=True,defect='B')

get_labels_defectA(path,class_number,num,exp=True,defect='AB',degrees=True)
get_labels_defectB(path,class_number,num,exp=True,defect='AB')

get_roi_rect(path,class_number,num,exp=True,defect='A')
get_roi_rect(path,class_number,num,exp=True,defect='B')
get_roi_rect(path,class_number,num,exp=True,defect='AB')


#DIBUJAR ALGUNOS AQUI PARA COMPROBAR LAS COSAS





