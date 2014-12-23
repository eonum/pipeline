package ch.eonum.pipeline.evaluation;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;

/**
 * Root mean square error.
 * 
 * @author tim
 *
 */
public class RMSE<E extends Instance> implements Evaluator<E> {

	public RMSE(){}

	@Override
	public double evaluate(DataSet<E> dataset) {
		return this.calculateEpsilon(dataset);
	}

	private double calculateEpsilon(DataSet<E> dataset) {
		double epsilon = 0.0;
		int n = 0;
		for(Instance inst : dataset){
			n++;
			epsilon += Math.pow(inst.getResult("result") - inst.outcome, 2.0);
		}
		epsilon /= n;
		epsilon = Math.sqrt(epsilon);
		return -epsilon;
	}

	@Override
	public void printResults(String fileName) {}

	@Override
	public void printResultsAndGnuplot(String fileName) {}

}
