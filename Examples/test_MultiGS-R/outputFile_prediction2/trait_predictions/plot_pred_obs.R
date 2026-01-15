args <- commandArgs(trailingOnly=TRUE)
if(length(args) < 3) stop('Usage: Rscript plot_pred_obs.R input.tsv output.png "Title" [width height]')
infile <- args[1]
outfile <- args[2]
title <- args[3]
width <- ifelse(length(args) >= 4, as.integer(args[4]), 800)
height <- ifelse(length(args) >= 5, as.integer(args[5]), 600)
dat <- try(read.table(infile, header=TRUE, sep='\t', stringsAsFactors=FALSE), silent=TRUE)
if(inherits(dat, 'try-error')) stop(paste('Cannot read input file:', infile))
if(!('observed' %in% colnames(dat)) || !('predicted' %in% colnames(dat))) stop('Input must have columns: observed and predicted')
x <- as.numeric(dat$observed)
y <- as.numeric(dat$predicted)
ok <- which(!is.na(x) & !is.na(y))
x <- x[ok]
y <- y[ok]
png(filename=outfile, width=width, height=height)
par(mar=c(5,4,4,2)+0.1)
if(length(x) >= 1) {
  plot(x, y, pch=19, xlab='Observed', ylab='Predicted', main=title)
  if(length(x) >= 2) {
    fit <- lm(y ~ x)
    abline(fit, col='red')
    r <- cor(x, y)
  } else { r <- NA }
  legend('topleft', legend=paste('Pearson R =', format(r, digits=4)), bty='n')
} else { plot.new(); text(0.5,0.5, 'No data to plot') }
dev.off()
