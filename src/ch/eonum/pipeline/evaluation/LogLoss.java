package ch.eonum.pipeline.evaluation;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;

/**
 * Log loss evaluation metric.
 * @author tim
 *
 */
public class LogLoss<E extends Instance> implements Evaluator<E> {

	@Override
	public double evaluate(DataSet<E> dataset) {
		double sum = 0.0;
		for(Instance each : dataset){
			double posteriori = each.getResult("result");
			sum += each.outcome * Math.log(posteriori) + 
					(1 - each.outcome) * Math.log(1 - posteriori);
		}
		return sum/dataset.size();
	}

	@Override
	public void printResults(String fileName) {}

	@Override
	public void printResultsAndGnuplot(String fileName) {}

}
