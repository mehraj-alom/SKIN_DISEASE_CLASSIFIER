# src/classifier/utils/class_mapping.py

cls_to_idx = {
 'Acne': 0,
 'Actinic keratosis': 1,
 'Atopic_Dermatitis': 2,
 'Bullous_Disease_Photos': 3,
 'Chickenpox': 4,
 'Dermatitis': 5,
 'Dermatofibroma': 6,
 'Dry_Skin': 7,
 'Eczema': 8,
 'Exanthems_And_Drug_Eruptions': 9,
 'Hair_Loss_Photos_Alopecia_And_Other_Hair_Diseases': 10,
 'Herpes': 11,
 'Hidradenitis-Suppurativa': 12,
 'Light_Diseases_And_Disorders_Of_Pigmentation': 13,
 'Lupus_And_Other_Connective_Tissue_Diseases': 14,
 'Nail Fungus And Other Nail Disease': 15,
 'Oily_Skin': 16,
 'Other diseases': 17,
 'Pa_Cutaneous_Larva_Migrans': 18,
 'Poison_Ivy_Photos_And_Other_Contact_Dermatitis': 19,
 'Psoriasis_Pictures_Lichen_Planus_And_Related_Diseases': 20,
 'Rashes': 21,
 'Rosacea': 22,
 'Shingles': 23,
 'Urticaria_Hives': 24,
 'Vitiligo': 25,
 'Warts': 26
}
all_labels = [k for k, v in sorted(cls_to_idx.items(), key=lambda item: item[1])]