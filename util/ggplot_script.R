# SCATTERPLOT OF BINARY

library(ggplot2)
to.read = file("/Users/franz/PycharmProjects/LearnedIndexes/data/datasets/Lognormal/lognormal.sorted.190M", "rb")
data=readBin(to.read, integer(), n = 190000000, size = 4, endian = "little")
colnames(data) = "values"

ys=data[1000000:1000100]
df <- data.frame(matrix(unlist(ys), nrow=length(ys), byrow=T))
gg<-ggplot(df, aes(x=ys,y=1000000:1000100))
gg+geom_point() + labs(title ="CDF Lognormal dataset", x = "Key", y = "Index")

df<-data.frame(matrix(unlist(data), nrow=length(data), byrow=T))
gg<-ggplot(df,aes(x=data,y=1:190000000))
gg+geom_smooth() + labs(title ="CDF Lognormal dataset", x = "Key", y = "Index")



# HEATMAP

library(ggplot2)
data = read.csv("/Users/franz/PycharmProjects/LearnedIndexes/data/predictions/recursive_learned_index/Integers_100x10x100k/new_reads.csv")
gg<-ggplot(data,aes(x=X0,y=X17.023))
ID=0:9
gg+geom_bin2d() + labs(title ="Data accesses for recursive learned index by entropy", x = "Entropy level", y = "Number of reads") + scale_x_continuous(breaks = ID)


library(ggplot2)
data = read.csv("/Users/franz/PycharmProjects/LearnedIndexes/data/predictions/naive_learned_index/Integers_100x10x100k/new_reads.csv")
ID=0:9
gg<-ggplot(data,aes(x=X0,y=X2.316))
gg+geom_bin2d() + labs(title ="Data accesses for naive learned index by entropy", x = "Entropy level", y = "Number of reads") + scale_x_continuous(breaks = ID)


# SCATTER RWD

library(ggplot2)
data = read.csv("/Users/franz/PycharmProjects/LearnedIndexes/data/datasets/Creditcard_285k/creditcard.csv_training")
gg<-ggplot(data,aes(x=X0.1,y=X0))
gg+geom_smooth() + labs(title ="CDF Creditcard dataset", x = "Key", y = "Index")