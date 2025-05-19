import json

base_thing_dataset_id_to_contiguous_id = {"0": 0, "1": 1, "3": 3, "4": 4, "5": 5, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29, "30": 30, "31": 31, "32": 32, "33": 33, "34": 34, "35": 35, "36": 36, "37": 37, "38": 38, "39": 39, "40": 40, "42": 42, "43": 43, "44": 44, "45": 45, "46": 46, "47": 47, "48": 48, "49": 49, "52": 52, "53": 53, "57": 57, "61": 61,}
novel_thing_dataset_id_to_contiguous_id = {"2": 2, "6": 6, "7": 7, "41": 41, "50": 50, "51": 51, "54": 54, "55": 55, "56": 56, "58": 58, "59": 59, "60": 60, "62": 62, "63": 63, "64": 64, "65": 65, "66": 66, "67": 67, "68": 68, "69": 69, "70": 70, "71": 71, "72": 72, "73": 73, "74": 74, "75": 75, "76": 76, "77": 77, "78": 78, "79": 79, "80": 80, "81": 81, "82": 82, "83": 83, "84": 84, "85": 85, "86": 86, "87": 87, "88": 88, "89": 89, "90": 90, "91": 91, "92": 92, "94": 93, "95": 94, "96": 95, "97": 96, "98": 97,
        "99": 98,
        "100": 99,
        "101": 100,
        "102": 101,
        "103": 102,
        "104": 103,
        "105": 104,
        "106": 105,
        "107": 106,
        "108": 107,
        "109": 108,
        "110": 109,
        "111": 110,
        "112": 111,
        "113": 112,
        "114": 113,
        "115": 114,
        "116": 115,
        "117": 116,
        "118": 117,
        "119": 118,
        "120": 119,
        "121": 120,
        "122": 121,
        "123": 122,
        "124": 123,
        "125": 124,
        "126": 125,
        "127": 126,
        "128": 127,
        "129": 128,
        "130": 129,
        "131": 130,
        "132": 131,
        "133": 132,
        "134": 133,
        "135": 134,
        "136": 135,
        "137": 136,
        "138": 137,
        "139": 138,
        "140": 139,
        "141": 140,
        "142": 141,
        "143": 142,
        "144": 143,
        "145": 144,
        "146": 145,
        "147": 146,
        "148": 147,
        "149": 148,
        "150": 149,
        "151": 150,
        "152": 151,
        "153": 152,
        "154": 153,
        "155": 154,
        "156": 155,
        "157": 156,
        "158": 157,
        "159": 158,
        "160": 159,
        "161": 160,
        "162": 161,
        "163": 162,
        "164": 163,
        "165": 164,
        "166": 165,
        "167": 166,
        "168": 167,
        "169": 168,
        "170": 169,
        "171": 170,
        "172": 171,
        "173": 172,
        "174": 173,
        "175": 174,
        "176": 175,
        "177": 176,
        "178": 177,
        "179": 178,
        "180": 179,
        "181": 180,
        "182": 181,
        "183": 182,
        "184": 183,
        "185": 184,
        "186": 185,
        "187": 186,
        "188": 187,
        "189": 188,
        "190": 189,
        "191": 190,
        "192": 191,
        "193": 192,
        "194": 193,
        "195": 194,
        "196": 195,
        "197": 196,
        "198": 197,
        "199": 198,
        "200": 199,
        "201": 200,
        "202": 201,
        "203": 202,
        "204": 203,
        "205": 204,
        "206": 205,
        "207": 206,
        "208": 207,
        "209": 208,
        "210": 209,
        "211": 210,
        "212": 211,
        "213": 212,
        "214": 213,
        "215": 214,
        "216": 215,
        "217": 216,
        "218": 217,
        "219": 218,
        "220": 219,
        "221": 220,
        "222": 221,
        "223": 222,
        "224": 223,
        "225": 224,
        "226": 225,
        "227": 226,
        "228": 227,
        "229": 228,
        "230": 229,
        "231": 230,
        "232": 231,
        "233": 232,
        "234": 233,
        "235": 234,
        "236": 235,
        "237": 236,
        "238": 237,
        "239": 238,
        "240": 239}

pred_mode = True
index = 7
datasets = ['Cityscapes3D_test', 'ARKitScenes_test', 'KITTI_test', 'nuScenes_test', 'Objectron_test', 'SUNRGBD_test', 'Hypersim_test', 'Waymo_test']
chosen_dataset = datasets[index]


with open(f'/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/{chosen_dataset}.json', 'rb') as f:
    data = json.load(f)
if pred_mode:
    with open(f'/cpfs01/user/zhanghanxue/omni3d/output/evaluation/inference/iter_final/{chosen_dataset}/omni_instances_results.json', 'rb') as f:
        pred_data = json.load(f)

imageid2anno = {}
for anno in data['annotations']:
    if anno['image_id'] in anno:
        imageid2anno[anno['image_id']].append(anno)
    else:
        imageid2anno[anno['image_id']] = [anno]
if pred_mode:
    pred_imageid2anno = {}
    for anno in pred_data:
        if anno['image_id'] in pred_imageid2anno:
            pred_imageid2anno[anno['image_id']].append(anno)
        else:
            pred_imageid2anno[anno['image_id']] = [anno]


oracle_list = []
for img in data['images']:
    sample = {}
    sample['image_id'] = img['id']
    sample['K'] = img['K']
    sample['width'] = img['width']
    sample['height'] = img['height']
    sample['instances'] = []
    if pred_mode:
        if img['id'] in pred_imageid2anno:

            for pred in pred_imageid2anno[img['id']]:
                # if pred['image_id'] == img['id']:
                sample['instances'].append({
                    'bbox': pred["bbox"],
                    'score': pred["score"],
                    'category_id': base_thing_dataset_id_to_contiguous_id[str(pred['category_id'])],
                    'category_name': None,
                })
        # oracle_list.append(sample)
        # pass
    else:
        for anno in imageid2anno:
            if anno['image_id'] == img['id'] and str(anno['category_id']) in base_thing_dataset_id_to_contiguous_id.keys():
                
                sample['instances'].append({
                    'bbox': [anno['bbox2D_proj'][0], anno['bbox2D_proj'][1], anno['bbox2D_proj'][2] - anno['bbox2D_proj'][0], anno['bbox2D_proj'][3] - anno['bbox2D_proj'][1]],
                    'score': 1,
                    'category_id': base_thing_dataset_id_to_contiguous_id[str(anno['category_id'])],
                    'category_name': anno['category_name'],
                })

    oracle_list.append(sample)

base_datasets = {
    'SUNRGBD_test': 'sunrgbd',
    'Hypersim_test': 'hypersim',
    'ARKitScenes_test': 'arkitscenes',
    'Objectron_test': 'objectron',
    'KITTI_test': 'kitti',
    'nuScenes_test': 'nuscenes',
    'Cityscapes3D_test': 'cityscapes3d',
    'Waymo_test': 'waymo',
    '3RScan_test': '3rscan',
}
if pred_mode:
    with open(f'cubercnn_{base_datasets[chosen_dataset]}_base_oracle_2d.json', 'w') as f:
        json.dump(oracle_list, f, indent=4)
else:
    with open(f'gt_{base_datasets[chosen_dataset]}_base_oracle_2d.json', 'w') as f:
        json.dump(oracle_list, f, indent=4)

