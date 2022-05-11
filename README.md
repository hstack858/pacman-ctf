# pacman-ctf
Final Project for CS4100 at Northeastern

---

### Instructions To Run
In order to run our team you can run the following command in this project's directory from your terminal:
```
python capture.py -r paccers -b {OTHER_TEAM_NAME} -l {MAP_NAME}
```

The names of the **teams** you can choose from are:

The names of the **maps** you can choose from are:
* alleyCapture
* bloxCapture
* crowdedCapture
* defaultCapture
* distantCapture
* fastCapture
* jumboCapture
* mediumCapture
* officeCapture
* strategicCapture
* testCapture
* tinyCapture

---
### Team Configuration
Due to the way the game's win condition is configured, I found that our team performs best with 2 offense agents and no defense agents. Our team is configured to run with both agent types, but you can switch to the optimal team by going to */paccers.py* and switching this line in the createTeam method 
**from:**
```
def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent', **args):
```
**to:**
```
def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='OffensiveAgent', **args):
```
