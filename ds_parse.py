import sys

def update_progress(current, total=None, prefix=''):
    if total:
        barLength = 50 # Length of the progress bar
        progress = current/total
        block = int(barLength*progress)
        text = "\r{}Progress: [{}] {:.1f}%".format(prefix, "#"*block + "-"*(barLength-block), progress*100)
    else:
        text = "\r{}Iter: {}".format(prefix, current)
    sys.stdout.write(text)
    sys.stdout.flush()
    return len(text)-1      # return length of text string not counting the initial \r

def json_cooked(x):
    #################################
    # Optimized version based on expected structure:
    # {"_label_cost":0,"_label_probability":0.01818182,"_label_Action":9,"_labelIndex":8,"Timestamp":"2017-10-24T00:00:15.5160000Z","Version":"1","EventId":"fa68cd9a71764118a635fd3d7a908634","a":[9,11,3,1,6,4,10,5,7,8,2],"c":{"_synthetic":false,"User":{"_age":0},"Geo":{"country":"United States","_countrycf":"8","state":"New York","city":"Springfield Gardens","_citycf":"8","dma":"501"},"MRefer":{"referer":"http://www.complex.com/"},"OUserAgent":{"_ua":"Mozilla/5.0 (iPad; CPU OS 10_3_2 like Mac OS X) AppleWebKit/603.2.4 (KHTML, like Gecko) Version/10.0 Mobile/14F89 Safari/602.1","_DeviceBrand":"Apple","_DeviceFamily":"iPad","_DeviceIsSpider":false,"_DeviceModel":"iPad","_OSFamily":"iOS","_OSMajor":"10","_OSPatch":"2","DeviceType":"Tablet"},"_multi":[{"
    # {"_label_cost":0,"_label_probability":1,"_label_Action":1,"_labelIndex":0,"_deferred":true,"Timestamp":"2018-10-25T00:01:31.1780000Z","Version":"1","EventId":"28EF7EE0B9CF4E319696CB812973F0B3","DeferredAction":true,"a":[1],"c":{"Global":{"SLOT":"1"},"Request":{"DISPLOC":"BR"},"Profile":{"_REQASID":"F2A3670E24F94646983BEB3EB35CB0C9","COUNTRY":"BR","FPIGM":"FALSE","FPNIGM":"FALSE","FB":"FALSE"},"_multi":[{"Action":{"constant":1,"PayloadID":"425039838"}}]},"p":[1.000000],"VWState":{"m":"DF30E6A3648947E69EE6B0816BF42640/1A5CD300DFE546B7B29A11EB70980809"}}
    # {"_label_cost":0,"_label_probability":0.833333015,"_label_Action":6,"_labelIndex":5,"o":[{"EventId":"A3E5ADF82D3A4BD5A5161FFC19C95DBB","DeferredAction":false}],"Timestamp":"2018-10-25T00:00:00.3960000Z","Version":"1","EventId":"A3E5ADF82D3A4BD5A5161FFC19C95DBB","DeferredAction":true,"a":[6,2,4,5,1,3],"c":{"Global":{"SLOT":"2"},"Request":{"DISPLOC":"US"},"Profile":{"_REQASID":"70A03B1FD0CA4B5A89669637DD448161","COUNTRY":"US","FPIGM":"FALSE","FPNIGM":"FALSE","FB":"FALSE","F1":0.0,"F2":0.0,"F3":0.00157480315,"F4":0.00182481752,"F5":0.0,"F6":0.00136892539,"F7":0.0,"F8":-1.0,"F9":0.0434782609,"F11":0.0,"F12":0.0,"F13":0.0,"F14":1.0,"F15":1.0,"F16":0.0,"F17":2.0,"F18":0.0,"F19":0.0,"F20":1.0
    # Assumption: "Version" value is 1 digit string
    #
    # Performance: 4x faster than Python JSON parser js = json.loads(x.strip())
    #################################
    ind1 = x.find(b',',16)              # equal to: x.find(',"_label_prob',16)
    ind2 = x.find(b',',ind1+23)         # equal to: x.find(',"_label_Action',ind1+23)
    ind3 = x.find(b',"T',ind2+34)       # equal to: x.find(',"Timestamp',ind2+34)
    ind7 = x.find(b',"a"',ind3+60)
    ind8 = x.find(b']',ind7+7)          # equal to: x.find('],"c',ind7+8)

    data = {}
    data['cost'] = x[15:ind1]                   # len('{"_label_cost":') = 15
    data['p'] = float(x[ind1+22:ind2])          # len(',"_label_probability":') = 22
    data['a_vec'] = x[ind7+6:ind8].split(b',')  # len(',"a":[') = 6
    data['a'] = int(data['a_vec'][0])
    data['num_a'] = len(data['a_vec'])
    data['skipLearn'] = b'"_skipLearn":true' in x[ind2+34:ind3] # len('"_label_Action":1,"_labelIndex":0,') = 34
            
    return data