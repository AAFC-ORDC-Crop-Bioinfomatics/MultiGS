args <- commandArgs(trailingOnly=TRUE)
infile <- args[1]; outfile <- args[2]; trait <- args[3]
d <- tryCatch(read.table(infile, header=TRUE, sep='\t', stringsAsFactors=FALSE), error=function(e) NULL)
if (is.null(d) || ncol(d) < 2) quit(status=1)
x <- suppressWarnings(as.numeric(d[[2]]))
x <- x[is.finite(x)]
if (length(x) == 0) quit(status=2)
png(outfile, width=1200, height=900, res=150)
par(mar=c(5,5,4,2)+0.1)
hist(x, breaks='FD', freq=FALSE, col='gray85', border='white', main=paste(trait, '(z-score)'), xlab=trait)
m <- mean(x); s <- sd(x); if (!is.finite(s) || s==0) s <- 1
curve(dnorm(x, mean=m, sd=s), add=TRUE, lwd=2)
dev.off()
