package ch.eonum.pipeline.evaluation;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Gnuplot;


/**
 * print a histogram for the two class problem. probability measures have to be
 * provided. some metrics are written to metrics.txt. This evaluator cannot be
 * used in optimization tasks as it does not give you a single value. It is
 * meant to be used for additional manual analysis.
 * 
 * @author tim
 * 
 */
public class Histograms<E extends Instance> implements Evaluator<E>  {

	protected DataSet<E> dataset;
	protected double max = Double.NEGATIVE_INFINITY;
	protected double min = Double.POSITIVE_INFINITY;

	@Override
	public double evaluate(DataSet<E> dataset) {
		this.dataset = dataset;
		this.calculateMinMax();
		return 0;
	}

	@Override
	public void printResults(String folderName) {
		int i = 0;
		int[] likelihoodHistogram = new int[600];
		int i_p = 0;
		int[] likelihoodHistogram_p = new int[600];
		int i_np = 0;
		int[] likelihoodHistogram_np = new int[600];
		double meanLikelihood = 0;
		double meanLikelihood_p = 0;
		double meanLikelihood_np = 0;
		for (Instance inst : dataset) {
			double likelihood = inst.getResult("result");

			i++;
			likelihoodHistogram[(int) Math.max(0, Math
					.min(599, Math.round(likelihood * 100)+300))]++;
			meanLikelihood += likelihood;
			if ("1".equals(inst.groundTruth)) {
				i_p++;
				likelihoodHistogram_p[(int) Math.max(0, Math
						.min(599, Math.round(likelihood * 100)+300))]++;
				meanLikelihood_p += likelihood;
			} else {
				i_np++;
				likelihoodHistogram_np[(int) Math.max(0, Math
						.min(599, Math.round(likelihood * 100)+300))]++;
				meanLikelihood_np += likelihood;
			}
		}

		meanLikelihood /= (double) i;
		meanLikelihood_p /= (double) i_p;
		meanLikelihood_np /= (double) i_np;

		double varianceLikelihood = 0.0;
		double varianceLikelihood_p = 0.0;
		double varianceLikelihood_np = 0.0;
		// write histograms
		try {
			PrintStream p = new PrintStream(new FileOutputStream(folderName
					+ "likelihoodZHistogram.txt"));

			for (int j = 0; j < likelihoodHistogram.length; j++) {
				p.println(((j - 300) / 100.0) + "	" + likelihoodHistogram[j]);
				// calculate variance
				varianceLikelihood += likelihoodHistogram[j]
						* Math.pow((meanLikelihood - (j - 300) / 100.0), 2);
			}
			
			p.close();
			p = new PrintStream(new FileOutputStream(folderName
					+ "likelihoodZHistogram_p.txt"));

			for (int j = 0; j < likelihoodHistogram_p.length; j++) {
				p.println(((j - 300) / 100.0) + "	" + likelihoodHistogram_p[j]);
				// calculate variance
				varianceLikelihood_p += likelihoodHistogram_p[j]
						* Math.pow((meanLikelihood_p - (j - 300) / 100.0), 2);
			}
			
			p.close();
			p = new PrintStream(new FileOutputStream(folderName
					+ "likelihoodZHistogram_np.txt"));

			for (int j = 0; j < likelihoodHistogram_np.length; j++) {
				p.println(((j - 300) / 100.0) + "	" + likelihoodHistogram_np[j]);
				// calculate variance
				varianceLikelihood_np += likelihoodHistogram_np[j]
						* Math.pow((meanLikelihood_np - (j - 300) / 100.0), 2);
			}
			
			varianceLikelihood /= (double) i;
			varianceLikelihood_p /= (double) i_p;
			varianceLikelihood_np /= (double) i_np;
			
			p.close();
			p = new PrintStream(new FileOutputStream(folderName
					+ "metrics.txt"));
			
			p.println();
			p.println("=== Total ===");
			p.println("Number of cases: " + i);
			p.println("Mean Likelihood: " + meanLikelihood);
			p.println("Variance likelihood: " + varianceLikelihood);
			p.println("Standard deviation likelihood: "
					+ Math.sqrt(varianceLikelihood));
			p.println("Maximum: " + max);
			p.println("Minimum: " + min);
			
			p.println();
			p.println("=== Plausible Cases ===");
			p.println("Number of cases: " + i_p);
			p.println("Mean Likelihood: " + meanLikelihood_p);
			p.println("Variance likelihood: " + varianceLikelihood_p);
			p.println("Standard deviation likelihood: "
					+ Math.sqrt(varianceLikelihood_p));
			
			p.println();
			p.println("=== Unplausible Cases ===");
			p.println("Number of cases: " + i_np);
			p.println("Mean Likelihood: " + meanLikelihood_np);
			p.println("Variance likelihood: " + varianceLikelihood_np);
			p.println("Standard deviation likelihood: "
					+ Math.sqrt(varianceLikelihood_np));
			
			p.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	protected void calculateMinMax() {
		for (Instance inst : dataset) {
			double value = inst.getResult("result");
			if (value > max)
				max = value;
			if (value < min)
				min = value;
		}
	}
	
	@Override
	public void printResultsAndGnuplot(String dir) {
		this.printResults(dir);
		Gnuplot.plotHistogram(new String[]{
				dir + "likelihoodZHistogram.txt", 
				dir + "likelihoodZHistogram_np.txt"}, dir + "histograms.png");
	}

}
