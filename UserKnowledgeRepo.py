import json
import os

class UserKnowledgeRepo():
    def __init__(self,filepath="user_profile.json"):
       self.filepath=filepath
       #Dacă fișierul nu există încă, îl creăm cu o listă goală
       if not os.path.exists(self.filepath):
           with open(self.filepath,"w",encoding="utf-8") as f:
               json.dump([],f)


    def get_knowledge(self, query=None):
        "Reads all the pacient history and format it as text for Advisor"
        if not os.path.exists(self.filepath):
            return "No available data for the patient"
        with open(self.filepath, "r",encoding="utf-8") as f:
            data=json.load(f)

        if not data:
            return "No available data for the patient"

        #Returnăm datele sub forma unei liste cu liniuță
        return "\n".join(f"-{item}" for item in data)

    def save_knowledge(self, new_info:str):
        "Save new information in the patient profile"
        #Evităm salvarea dacă modelul a spus că nu a găsit date utile
        if not new_info or new_info.strip()=="" or "No new medical data" in new_info:
            return
        with open(self.filepath,"r",encoding="utf-8") as f:
            data=json.load(f)

        #Adăugăm noua informație extrasă la lista existentă
        data.append(new_info.strip())

        with open(self.filepath,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=4)