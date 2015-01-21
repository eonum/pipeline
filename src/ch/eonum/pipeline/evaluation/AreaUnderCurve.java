package ch.eonum.pipeline.evaluation;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;


/**
 * Calculate the area under curve for a two-class problem. Preconditions: Two
 * classes: "1" and "0". The "result" result should indicate the probability of
 * class "1".
 * 
 * @author tim
 * 
 */
public class AreaUnderCurve<E extends Instance> implements Evaluator<E> {

	protected DataSet<E> dataset;
	protected double max = Double.NEGATIVE_INFINITY;
	protected double min = Double.POSITIVE_INFINITY;
	private double auc;
	private List<String> printList;

	@Override
	public double evaluate(DataSet<E> dataset) {
		this.dataset = dataset;
		calculateMinMax();
		calculateAUC();
		return this.auc ;
	}
	
	@Override
	public void printResults(String fileName) {
		try {
			FileWriter fstream = new FileWriter(fileName);
			BufferedWriter out = new BufferedWriter(fstream);
			for(String each : this.printList){
				out.write(each + "\n");
			}
			out.close();
		} catch (Exception e) {
			Log.warn(e.getMessage());
		}
	}

	/**
	 * calculate the are under curve.
	 */
	private void calculateAUC() {
		this.auc = 0.0;
		this.printList = new ArrayList<String>();
		double lastx = 1.0;
		double lasty = 1.0;
		/** iterate over all likelihood values using 400 steps. */
		for (int i = -1; i < 401; i++) {
			double threshold = min + (max - min) * (i / 400.0);
			int true_positive = 0;
			int true_negative = 0;
			int false_positive = 0;
			int false_negative = 0;
			for (Instance inst : dataset) {
				double result = (inst.label == null || inst.label.equals("0")) ? inst
						.getResult("result") : -inst.getResult("result");
				if (result > threshold
						&& "0".equals(inst.groundTruth))
					true_positive++;
				if (result < threshold
						&& "0".equals(inst.groundTruth))
					true_negative++;
				if (result > threshold
						&& "1".equals(inst.groundTruth))
					false_positive++;
				if (result < threshold
						&& "1".equals(inst.groundTruth))
					false_negative++;
			}

			double fr = (double) true_negative
					/ (true_negative + true_positive);
			double fa = (double) false_positive
					/ (false_negative + false_positive);
			double x = fa;
			double y = 1.0 - fr;
			this.printList.add(x + "\t" + y);
			
			auc += (lastx - x) * ((lasty + y)/2);
			lastx = x;
			lasty = y;
		}
	}

	/**
	 * Calculate the maximum and the minimum value for the output
	 * probability/likelihood.
	 */
	protected void calculateMinMax() {
		for (Instance inst : dataset) {
			double value = (inst.label == null || "0".equals(inst.label)) ? inst.getResult("result")
					: -inst.getResult("result");
			if (value > max)
				max = value;
			if (value < min)
				min = value;
		}
	}

	@Override
	public void printResultsAndGnuplot(String fileName) {
		this.printResults(fileName);
		Gnuplot.plotAreaUnderCurve(fileName, "Area under Curve");
	}

}
