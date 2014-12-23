package ch.eonum.pipeline.evaluation;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Gnuplot;

public class RecognitionRate<E extends Instance> implements Evaluator<E> {
	/**
	 * number of k best ranked classes taking into account when doing k best
	 * oracle.
	 */
	private int k;
	private Map<Integer, Double> oracle;
	private Set<String> classes;
	
	public RecognitionRate(){
		this.k = 0;
	}

	@Override
	public double evaluate(DataSet<E> dataset) {
		int correct = 0;
		int wrong = 0;
		for(Instance each : dataset)
			if(each.groundTruth != null && each.groundTruth.equals(each.label))
				correct++;
			else
				wrong++;
		
		if(k > 0){
			dataset.calculateRanks(classes);
			this.oracle = new HashMap<Integer, Double>();
			for(int i = 0; i < k; i++)
				oracle.put(i, 0.);
			for(Instance each : dataset){
				double rank = each.getResult("rank");
				for(int i = 0; i < k; i++)
					if(rank <= i)
						oracle.put(i, oracle.get(i) + 1);
			}
			for(int i = 0; i < k; i++)
				oracle.put(i, oracle.get(i)/dataset.size());
		}
		
		return (double)correct/(correct+wrong);
	}

	@Override
	public void printResults(String fileName) {
		
	}

	@Override
	public void printResultsAndGnuplot(String fileName) {
		if(k > 0){
			this.printResults(fileName);
			Gnuplot.plotOneDimensionalCurve(oracle, "k-best oracle", fileName);
		}
	}
	
	public void doKBestOracle(int k, Set<String> classes){
		this.k = k;
		this.classes = classes;
	}

}
