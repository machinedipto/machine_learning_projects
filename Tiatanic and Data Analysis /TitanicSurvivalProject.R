train.tit = read.csv("/home/orienit/Downloads/Titanic Train.csv",stringsAsFactors = F)
dim(train.tit)
test.tit = read.csv("/home/orienit/Downloads/test (1).csv",stringsAsFactors = F)

test.tit$Survived = NA

full.titanic = rbind(train.tit,test.tit)

str(full.titanic)
attach(full.titanic)
#missing value imputation
colSums(is.na(full.titanic))
colSums(full.titanic =='')
# > colSums(is.na(full.titanic))
# PassengerId    Survived      Pclass        Name         Sex         Age 
# 0         418           0           0           0         263 
# SibSp       Parch      Ticket        Fare       Cabin    Embarked 
# 0           0           0           1           0           0 
###############################################################################
# PassengerId    Survived      Pclass        Name         Sex         Age 
# 0          NA           0           0           0          NA 
# SibSp       Parch      Ticket        Fare       Cabin    Embarked 
# 0           0           0          NA        1014           2 
full.titanic[full.titanic$Embarked == '',]
temp = full.titanic[full.titanic$Fare > 75,"Embarked"]
temp = as.factor(temp)

table(temp)

### as we can see the ticket fare which is greater than 75 has maximum frequency over Embarked C so we should impute the missing value with C

full.titanic$Embarked[full.titanic$Embarked == ''] = "C"

apply(full.titanic,2,function(x) {length(unique(x))})
# > apply(full.titanic,2,function(x) {length(unique(x))})
# PassengerId    Survived      Pclass        Name         Sex         Age       SibSp       Parch      Ticket 
# 1309           3           3        1307           2          99           7           8         929 
# Fare       Cabin    Embarked 
# 282         187           3 

full.titanic[is.na(full.titanic$Fare),]

pclem = full.titanic[Pclass==3 & Embarked == "S",]

median(pclem$Fare,na.rm = T)
###### we are going to impute the median value of 1044 passengers to 8.05 

full.titanic$Fare[PassengerId == 1044] = 8.05

######### we are only left with Age and Cabin as in cabin there are 1014 missing value we can't do much there.

cols = c ("Survived","Pclass","Sex","Embarked")
for (i in cols){
  full.titanic[,i]=as.factor(full.titanic[,i])
}

######feature engineering 
### Reach people survived we shown the same as movie also HAHA 

library(ggplot2)

ggplot(full.titanic[1:891,],aes(x = Pclass, fill = factor(Survived))) + geom_bar()

# we can see the 1st class survived more

# as Rose survived in the movie we want to check if there is a relation on sex and survival

ggplot(full.titanic[1:891,],aes(x = Sex, fill = factor(Survived))) + geom_bar() + facet_wrap(~Pclass)
 ## in the all clases female survived more 

#########Next we are entering into family dilema hopefully coming from a big family together might have a better possibilty or surving 
######also it can true that the family can dire together
full.titanic$familySize = full.titanic$SibSp + full.titanic$Parch +1

full.titanic$familySize

full.titanic$FamilySized[full.titanic$familySize == 1] = "Single"
full.titanic$FamilySized[full.titanic$familySize< 5 & full.titanic$familySize >=2] = "Small"
full.titanic$FamilySized[full.titanic$familySize>=5] = "Big" 

full.titanic$FamilySized = as.factor(full.titanic$FamilySized)
ggplot(full.titanic[1:891,],aes(x = FamilySized,fill = factor(Survived))) + geom_bar()

names.tit = full.titanic$Name
names.title = gsub("^.*, (.*?)\\..*$", "\\1", names.tit)

full.titanic$title = names.title

table(full.titanic$title)

################creating another variable using family ID which will help to correctly identify family using their surname and in panic large family might not be able to pick up the boat
full.titanic$Surname = sapply(full.titanic$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
full.titanic$FamilyID = paste(as.character(full.titanic$familySize), full.titanic$Surname, sep="")
full.titanic$FamilyID[full.titanic$familySize<=2] = 'Small'
#full.titanic$FamilyID[full.titanic$FamilyID == '3Peacock'] = 'Small'
table(full.titanic$FamilyID)
famIDs <- data.frame(table(full.titanic$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
full.titanic$FamilyID[full.titanic$FamilyID %in% famIDs$Var1] = 'Small'
full.titanic$FamilyID = factor(full.titanic$FamilyID)
table(full.titanic$FamilyID)

# > table(full.titanic$title)
# 
# Capt          Col          Don         Dona           Dr     Jonkheer 
# 1            4            1            1            8            1 
# Lady        Major       Master         Miss         Mlle          Mme 
# 1            2           61          260            2            1 
# Mr          Mrs           Ms          Rev          Sir the Countess 
# 757          197            2            8            1            1

full.titanic$title[full.titanic$title=='Dona'] = 'Miss'
full.titanic$title[full.titanic$title=='Lady'] = 'Miss'
full.titanic$title[full.titanic$title=='Mlle'] = 'Miss'
full.titanic$title[full.titanic$title=='Mme'] = 'Mrs'
full.titanic$title[full.titanic$title=='Ms'] = 'Mrs'

### as creating more variable among features can cause overfit we need get rid of the
### other qualitative measures like col Don into one category i name that officers 
### bcse as i know from movie also all the crew might sacrificed thier lifes

full.titanic$title[full.titanic$title == 'Capt'] = 'Officer' 
full.titanic$title[full.titanic$title == 'Col'] = 'Officer' 
full.titanic$title[full.titanic$title == 'Major'] = 'Officer'
full.titanic$title[full.titanic$title == 'Dr'] = 'Officer'
full.titanic$title[full.titanic$title == 'Rev'] = 'Officer'
full.titanic$title[full.titanic$title == 'Don'] = 'Officer'
full.titanic$title[full.titanic$title == 'Sir'] = 'Officer'
full.titanic$title[full.titanic$title == 'the Countess'] = 'Officer'
full.titanic$title[full.titanic$title == 'Jonkheer'] = 'Officer'

ggplot(full.titanic[1:891,],aes(x = title,fill=factor(Survived)))  + geom_bar()

ggplot(full.titanic[1:891,],aes(x= title , fill = factor(Survived))) + geom_bar() + facet_wrap(~Pclass)

ggplot(full.titanic[1:891,],aes(x = FamilySized,fill = factor(Survived))) + geom_bar() + facet_wrap(~title)

full.titanic$title = as.factor(full.titanic$title)


#######################Engineer features based on all the passengers with the same ticket
ticket.unique = rep(0, nrow(full.titanic))
tickets = unique(full.titanic$Ticket)

for (i in 1:length(tickets)) {
  current.ticket <- tickets[i]
  party.indexes <- which(full.titanic$Ticket == current.ticket)
  for (k in 1:length(party.indexes)) {
    ticket.unique[party.indexes[k]] <- length(party.indexes)
  }
}

ticket.unique
full.titanic$ticketUnique = ticket.unique

full.titanic$ticketSize[full.titanic$ticketUnique == 1] = 'Single'
full.titanic$ticketSize[full.titanic$ticketUnique < 5 & full.titanic$ticketUnique>=2] = 'Small'
full.titanic$ticketSize[full.titanic$ticketUnique>=5] = 'Big'
full.titanic$ticketSize = as.factor(full.titanic$ticketSize)
ggplot(full.titanic[1:891,],aes(x = ticketSize,fill = factor(Survived))) + geom_bar() + facet_wrap(~title)


#####################missing value inputation for Age 
library(mice)


library(randomForest)
mice_mod <- mice(full.titanic[, !names(full.titanic) %in% c('PassengerId','Name','Ticket','Cabin','title','Survived','FamilySized')], method='rf')

mice_output = complete(mice_mod)

mice_output

full.titanic$Age = mice_output$Age

full.titanic$Child = 'adult'
full.titanic$Child[full.titanic$Age<10] = 'kid'
full.titanic$Child[full.titanic$Age>80] = 'old'
full.titanic$Child = as.factor(full.titanic$Child)


ggplot(full.titanic[1:891,],aes(x= Child , fill = factor(Survived))) + geom_bar() + facet_wrap(~Pclass)

##### we can see most of the child survived

full.titanic$Mother <- 'Not Mother'

full.titanic$Mother[full.titanic$Sex =='female' & full.titanic$Parch > 0 & full.titanic$Age > 18 & full.titanic$title !='Miss'] = 'Mother'

md.pattern(full.titanic)
full.titanic$Mother = as.factor(full.titanic$Mother)

ggplot(full.titanic[1:891,],aes(x=Mother, fill = factor(Survived))) + geom_bar() + facet_wrap(~Pclass)

######### as the classification problems works good for qualitative variables i am creating the last variable Fare2 for different range of fare
######## hopefully the rich guys i mean who paid more got survived

ggplot(full.titanic[1:891,],aes(x = Pclass, y=Fare, fill = factor(Survived))) + geom_boxplot()
full.titanic$Fare2 = 'thirty+'
full.titanic$Fare2[full.titanic$Fare<30 & full.titanic$Fare>=20] = 'twenty+'
full.titanic$Fare2[full.titanic$Fare<20 & full.titanic$Fare>=10] = 'ten+'
full.titanic$Fare2[full.titanic$Fare<10] = 'ten-'

full.titanic$Fare2 = as.factor(full.titanic$Fare2)
ggplot(full.titanic[1:891,], aes(x = Fare2 , fill = factor(Survived))) + geom_bar() + facet_wrap(~Sex)
#########Runing logistic Model #############
names(full.titanic)
feature1 = full.titanic[1:891,c("Pclass","Sex","Embarked","FamilySized","title","ticketSize","FamilyID","Child","Mother","Fare2")]
response = as.factor(train.tit$Survived)
feature1$Survived = as.factor(train.tit$Survived)


actualtestfeature = full.titanic[892:1309,c("Pclass","Sex","Embarked","FamilySized","title","ticketSize","FamilyID","Child","Mother","Fare2")]
actualtestfeature$Survived = NA
levels(actualtestfeature$FamilyID)
levels(feature1$FamilyID)
###For Cross validation purpose will keep 20% of data aside from my orginal train set
##This is just to check how well my data works for unseen data
set.seed(500)
library(caret)
ind=createDataPartition(feature1$Survived,times=1,p=0.8,list=FALSE)
train_val=feature1[ind,]
test_val=feature1[-ind,]

####check the proprtion of Survival rate in orginal training data, current traing and testing data
round(prop.table(table(train.tit$Survived)*100),digits = 1)
# 0   1 
# 0.6 0.4 

round(prop.table(table(train_val$Survived)*100),digits = 1)
# 0   1 
# 0.6 0.4 

round(prop.table(table(train.tit$Survived)*100),digits = 1)
# 0   1 
# 0.6 0.4 

############## Similar proportion for all hopefully logistic predict it good 

contrasts(train_val$Sex)
contrasts(train_val$Pclass)
contrasts(train_val$Fare2)
contrasts(train_val$Child)
contrasts(train_val$Mother)
contrasts(train_val$Embarked)

glmfit = glm(Survived~.,family = binomial(link = logit),data = train_val)
summary(glmfit)

glmfit2 = glm(Survived~.,family = binomial(link = logit),data = feature1)
summary(glmfit2)
######lets predict the train data only 
trainprob = predict(glmfit,data = train_val, type = "response")
table(train_val$Survived,trainprob>0.5)

featureProb = predict(glmfit2,data = feature1,type = "response")
length(featureProb)
table(feature1$Survived,featureProb>0.5)
# > (395 + 200) / (74 + 45 + 395 + 200)
# [1] 0.8333333

testprob = predict(glmfit, newdata = test_val, type = "response")
table(test_val$Survived,testprob>0.5)

dim(actualtestfeature)
actualtestfeatureProb5 = predict(glmfit2,actualtestfeature,type = "response")
length(actualtestfeatureProb5)
actualtestfeature$Survived[actualtestfeatureProb5>0.5] = 1
actualtestfeature$Survived[actualtestfeatureProb5<=0.5] = 0
actualtestfeature$Survived

table(actualtestfeature$Survived)

test.tit$Survived = actualtestfeature$Survived

test.tit$Survived

###########applying KNN

testk = test_val[,c(1:8)]
traink = train_val[,c(1:8)]
testk = data.matrix(testk)
traink = data.matrix(traink)
summary(traink)

testSurvived = as.numeric(as.character(test_val$Survived))
as.matrix()
trainSurvived = as.numeric(as.character(train_val$Survived))
is.vector(trainSurvived)
knn.pred = knn(as.matrix(traink),as.matrix(testk),trainSurvived,k=2)
table(knn.pred,testSurvived)
mean(knn.pred == testSurvived)
#############actual submission knn ##########
featurek = feature1[,c(1:8)]
actualtestfeaturek = actualtestfeature[,c(1:8)]
featurek = data.matrix(featurek)
actualtestfeaturek = data.matrix(actualtestfeaturek)
trainSurvivedfeature = as.numeric(as.character(feature1$Survived))
knn.predactual = knn(featurek,actualtestfeaturek,trainSurvivedfeature,k = 2)
knn.predactual
table(knn.predactual)
test.tit$Survived = knn.predactual
test.tit$Survived

submitPassengerID = test.tit$PassengerId
submitSurvived = test.tit$Survived

submit = data.frame("PassengerID" = submitPassengerID,"Survived"=submitSurvived)
?write.csv
write.csv(submit,"TitanicSubmissionLogistic2.csv",row.names = F)
getwd()

############ getting random forest
install.packages('randomForest')
library(randomForest)


full.titanic$FamilyID2 = full.titanic$FamilyID
full.titanic$FamilyID2 = as.character(full.titanic$FamilyID2)
full.titanic$FamilyID2[full.titanic$familySize <=3] = 'Small'
full.titanic$FamilyID2 = as.factor(full.titanic$FamilyID2)
feature2 = full.titanic[1:891,c("Pclass","Sex","Embarked","FamilySized","title","Child","Fare","Mother","FamilyID2")]
response2 = as.factor(train.tit$Survived)
feature2$Survived = as.factor(train.tit$Survived)


actualtestfeature2 = full.titanic[892:1309,c("Pclass","Sex","Embarked","FamilySized","title","Child","Fare","Mother","FamilyID2")]
actualtestfeature2$Survived = NA
rfit = randomForest(Survived~.,data =feature2, importance = T, ntree =2000)

varImpPlot(rfit)

pred = predict(rfit,actualtestfeature2)
pred
submi1 = data.frame(PassengerId = test.tit$PassengerId, Survived = pred)
write.csv(submi1,"TitanicrandomForest.csv", row.names = F)
 library(party)
rfit2 = cforest(Survived~.,data = feature2,controls = cforest_unbiased(ntree=2000,mtry =3))
Prediction <- predict(rfit2,actualtestfeature2, OOB=TRUE, type = "response")
submi2 = data.frame(PassengerId = test.tit$PassengerId, Survived = Prediction)
write.csv(submi2,"TitanicrandomForest9.csv", row.names = F)
l