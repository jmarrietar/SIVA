# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:48:22 2017

@author: josemiguelarrieta
"""
#Load libraries
import os 
os.chdir('Documents/SIVA')

from utils_dagm import get_labels_defectA, get_labels_defectB, get_roi_rect,write_labels_defectB,write_labels_expROI,write_labels_defectA
from shutil import copyfile

#Que la lista se realice preguntandome cuales son las Imagenes que Voy a usar. 

#######################################################################################
# NOTA importante: Cambiar Esto de list_defect_AB /estoy quemando ese terminado en AB #
#######################################################################################

list_defect_AB = [1,2,3,4,5,6]

class_number = 1
defect = 'AB'
file_type = '.png'

path = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'
src = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/optical2/Class'+str(class_number)+'_def'+defect+'/'
dst = '/Users/josemiguelarrieta/Dropbox/11_Semestre/Jovenes_Investigadores/images/Experiments_DAGM/Class'+str(class_number)+'/Cl_'+str(class_number)+'_'+str(defect)+'/'

contador = 1
for i in range (0,len(list_defect_AB)):
    print i
    filename = str(list_defect_AB[i])+'_'+defect+file_type
    filename_dest = 'Cl_'+ str(class_number)+'_'+str(list_defect_AB[i])+'_'+defect+file_type
    copyfile(src+filename, dst+filename_dest)
    cord_defect_A = get_labels_defectA(path,class_number,list_defect_AB[i])
    cord_defect_B = get_labels_defectB(path,class_number,list_defect_AB[i])
    cord_roi = get_roi_rect(path,class_number,list_defect_AB[i])
    write_labels_defectA(class_number,list_defect_AB[i],cord_defect_A,reason='new_experiment',new_num=contador,defect=defect)
    write_labels_defectB(class_number,list_defect_AB[i],cord_defect_B['x1'],cord_defect_B['y1'],cord_defect_B['x2'],cord_defect_B['y2'],reason='new_experiment',new_num=contador,defect=defect)
    write_labels_expROI(class_number,list_defect_AB[i],cord_roi['x1'],cord_roi['y1'],cord_roi['x2'],cord_roi['y2'],reason='new_experiment',new_num=contador,defect=defect)
    contador+=1



