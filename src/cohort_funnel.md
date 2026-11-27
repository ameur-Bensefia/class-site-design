# Cohort selection funnel, reconstructed from the retained samples and the original metadata
root: `<CSD_DATA_ROOT>`; datasets: `<CSD_DATASETS_DIR>`

## NIH ChestX-ray14 (effusion vs no finding)

- metadata: `<CSD_DATA_ROOT>/NIH ChestX/Data_Entry_2017.csv`
- release: 112,120 images, 30,805 patients
- eligible: effusion (label list contains 'Effusion') 13,317 images; 'No Finding' 60,361 images
  - of the effusion images, with no other finding: 3,955; with co-occurring findings: 9,362
- retained: 20,000 images (3,632 label 1, 16,368 label 0); found in metadata: 20,000; patients 7,022
- rule check: label-1 images whose labels contain 'Effusion': 100.0%; label-0 images that are exactly 'No Finding': 100.0%
- retained effusion images with co-occurring findings: 2,549
- view position, retained: {'PA': 12328, 'AP': 7672}; eligible: {'PA': 45891, 'AP': 27787}
- subsampling: 7,022 of 25,923 eligible patients selected; 20,000 of the 20,000 eligible images of the selected patients retained (100.0%; 100% = whole patients kept)

## CheXpert (pleural effusion vs no finding)

- metadata: `<CSD_DATA_ROOT>/chexpert/train.csv` + valid.csv
- release (train and valid): 223,648 images, 64,740 patients
- frontal views: 191,229
- pleural effusion among frontal: positive 76,963, uncertain 9,578, negative 25,437, blank 79,251; 'No Finding' = 1: 17,000
- eligible (MedAudit rule): effusion = 1: 76,963; no finding = 1 with effusion neither 1 nor -1: 17,000; excluded because effusion uncertain: 9,578
- retained: 20,001 images (16,421 label 1, 3,580 label 0); found in metadata: 20,001; patients 7,841
- rule check: retained images that are frontal 100.0%
  - label 1: Pleural Effusion = 1 in 100.0%; label 0: No Finding = 1 in 100.0%, Pleural Effusion uncertain (-1) in 0, blank in 1,874
- AP/PA, retained: {'AP': 17156, 'PA': 2842, 'LL': 3}
- subsampling: 7,841 of 37,217 eligible patients selected; 20,001 of 20,001 eligible images of the selected patients retained (100.0%)

## HAM10000 (melanoma vs nevus)

- metadata: `<CSD_DATA_ROOT>/HAM10000/HAM10000_metadata`
- release: 10,015 images, 7,470 lesions; diagnoses {'nv': 6705, 'mel': 1113, 'bkl': 1099, 'bcc': 514, 'akiec': 327, 'vasc': 142, 'df': 115}
- eligible (mel or nv): 7,818 images (1,113 mel, 6,705 nv), 6,017 lesions
- retained: 7,818 images matched to metadata: 7,818; by source: {('rosendahl', 'mel'): 342, ('rosendahl', 'nv'): 803, ('vidir_modern', 'mel'): 680, ('vidir_modern', 'nv'): 1832, ('vidir_molemax', 'mel'): 24, ('vidir_molemax', 'nv'): 3720, ('vienna_dias', 'mel'): 67, ('vienna_dias', 'nv'): 350}
- .npz: 7,818 samples, label 1 = 1,113; sites {'vidir_molemax': 3744, 'vidir_modern': 2512, 'rosendahl': 1145, 'vienna_dias': 417}

## PTB-XL (myocardial infarction vs normal)

- .npz: 11,601 records, label 1 = 2,532, 10,652 patients, 52 sites
- metadata: `<CSD_DATA_ROOT>/ptb-xl/ptbxl_database.csv`
- release: 21,799 records, 18,869 patients
- eligible (MedAudit rule, single superclass): MI only 2,532, NORM only 9,069; excluded: MI with another superclass 2,937, other or no superclass 7,261
- diagnostic superclasses: MI present 5,469; NORM only 9,069; MI and NORM 1; no diagnostic statement 411
- rule check (retained): label 1 with MI 100.0%, of which also NORM 0; label 0 NORM only 100.0%; label 1 with other superclasses besides MI 0

## ABIDE I and II (autism vs control)

- .npz: 2,221 slices, label 1 = 1,058, 2,221 identifiers, 39 sites
- sites: {'ABIDEII-KKI_1': 209, 'NYU': 184, 'UM_1': 110, 'ABIDEII-GU_1': 105, 'USM': 101, 'ABIDEII-OHSU_1': 93, 'UCLA_1': 82, 'ABIDEII-NYU_1': 78, 'ABIDEII-BNI_1': 58, 'ABIDEII-SDSU_1': 58, 'ABIDEII-OILH_2': 58, 'MAX_MUN': 57, 'PITT': 57, 'YALE': 56, 'ABIDEII-IP_1': 56, 'KKI': 55, 'ABIDEII-EMC_1': 54, 'TRINITY': 49, 'ABIDEII-TCD_1': 42, 'ABIDEII-SU_2': 42, 'STANFORD': 40, 'ABIDEII-IU_1': 40, 'CALTECH': 38, 'ABIDEII-ETH_1': 37, 'OLIN': 36, 'SDSU': 36, 'LEUVEN_2': 35, 'UM_2': 35, 'ABIDEII-USM_1': 33, 'ABIDEII-UCLA_1': 32, 'ABIDEII-UCD_1': 32, 'SBL': 30, 'LEUVEN_1': 29, 'ABIDEII-KUL_3': 28, 'OHSU': 28, 'ABIDEII-U_MIA_1': 28, 'ABIDEII-NYU_2': 27, 'CMU': 27, 'UCLA_2': 26}
- phenotypic files found: ['ABIDEII_Composite_Phenotypic.csv', 'Phenotypic_V1_0b_preprocessed1.csv']
