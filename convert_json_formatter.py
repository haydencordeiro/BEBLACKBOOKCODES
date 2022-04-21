import json
d={}
def calc_state(val):
    states = [
        [0, 0.2],
        [0.2, 0.4],
        [0.4, 0.6],
        [0.6, 0.8],
        [0.8, 1],
    ]
    for i in range(len(states)):
        if (states[i][0] <= val and val < states[i][1]): 
            return i
    return 4

def string_state(val):
    return ["Not Interested", "Slightly Interested","Neutral","Slightly Interested","Highly Interested"][val]    
			

final_json=[]
for k,v in d.items():
    temp_bool=True
    for temp_k in ["deep","emo","orient","zfinal_score","zimage_url"]:
        if temp_k not in v:
            temp_bool = False
            continue
    if k.split(",")[0] !="2":continue
    if not temp_bool : continue
    temp = v
    temp["state"] = string_state(calc_state(v["zfinal_score"]))
    temp["zfinal_score"] = (round( temp["zfinal_score"], 2))
    temp["zimage_url"] = "videoserver/"+v["zimage_url"].split("/")[-1]
    final_json.append(temp)
print(json.dumps({"data":final_json}, indent = 4))