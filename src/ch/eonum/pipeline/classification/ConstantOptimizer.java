package ch.eonum.pipeline.classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.evaluation.Evaluator;

/**
 * <p>
 * Regressor
 * </p>
 * <p>Simple search for an optimal grid which is separated by constants on X
 * dimensions, where X is the number of limit arrays. For each distinct cell an
 * optimized constant is assigned to a tested instance.</p>
 * 
 * <p>This is a very simple regressor used mainly for benchmarking purposes.</p>
 * 
 * @author tim
 * 
 * @param <E>
 */
public class ConstantOptimizer<E extends Instance> extends Classifier<E> {
	private Evaluator<E> evaluator;
	private Map<String, Double[]> limits;
	/** overall optimum. */
	private double optConstant;
	private Map<Map<String, Range>, Double> constants;
	private List<Map<String, Range>> combinations;

	public ConstantOptimizer(Evaluator<E> evaluator, Map<String, Double[]> limits) {
		this.evaluator = evaluator;
		this.limits = limits;
		this.constants = new HashMap<Map<String, Range>, Double>();
	}

	@Override
	public void train() {
		double maxE = Double.NEGATIVE_INFINITY;
		optConstant = 0.0;
		int i = 0;
		int stepsSinceLastImprovement = 0;
		while(true){
			stepsSinceLastImprovement++;
			double constant = (double)(i/100.0);
			i++;
			for(Instance each : this.trainingDataSet)
				each.putResult("result", constant);
			double e = this.evaluator.evaluate(trainingDataSet);
			if(maxE < e){
				maxE = e;
				optConstant = constant;
				stepsSinceLastImprovement = 0;
			}
			if(stepsSinceLastImprovement > 10)
				break;
		}
		System.out.println("Optimum for Overall Constant = " + maxE + " Constant = " + optConstant);
		
		for(Instance each : trainingDataSet)
			each.putResult("result", optConstant);
		
		this.combinations = this.getAllCombinations(new HashMap<String, Double[]>(this.limits));
		
		for(Map<String, Range> combination : combinations){
			maxE = Double.NEGATIVE_INFINITY;
			double optConstantF = 0.0;
			i = 0;
			stepsSinceLastImprovement = 0;
			while(true){
				stepsSinceLastImprovement++;
				double constant = (double)(i/100.0);
				i++;
				for(Instance each : this.trainingDataSet){
					if(this.getCondition(each, combination))
						each.put("result", constant);
				}
				double e = this.evaluator.evaluate(trainingDataSet);
				if(maxE < e){
					maxE = e;
					optConstantF = constant;
					stepsSinceLastImprovement = 0;
				}
				if(stepsSinceLastImprovement > 10)
					break;
			}
			for(Instance each : trainingDataSet)
				if(this.getCondition(each, combination))
					each.putResult("result", optConstantF);
			
			this.constants.put(combination, optConstantF);
			System.out.println("Optimum for constant (" + combination
					+ ") = " + maxE + " Constant = " + optConstantF);
		}
	}
	
	private boolean getCondition(Instance each, Map<String, Range> combination) {
		boolean condition = true;
		for (String feature : combination.keySet()) {
			double v = each.get(feature);
			condition = condition && combination.get(feature).end > v;
			condition = condition && combination.get(feature).start <= v;
		}
		return condition;
	}

	private List<Map<String, Range>> getAllCombinations(
			Map<String, Double[]> restLimit) {
		
		if(restLimit.isEmpty()){
			ArrayList<Map<String, Range>> list = new ArrayList<Map<String, Range>>();
			list.add(new HashMap<String, Range>());
			return list;
		}
			
		
		String feature1 = restLimit.keySet().iterator().next();
		Double[] ra = restLimit.get(feature1);
		double[] ranges = new double[ra.length + 2];
		ranges[0] = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < ra.length; i++)
			ranges[i+1] = ra[i];
		ranges[ranges.length - 1] = Double.POSITIVE_INFINITY;
		restLimit.remove(feature1);
		
		List<Map<String, Range>> list = new ArrayList<Map<String, Range>>();
		for(int i = 0; i < ranges.length - 1; i++){
			List<Map<String, Range>> list2 = this.getAllCombinations(new HashMap<String, Double[]>(restLimit));
			for(Map<String, Range> each : list2)
				each.put(feature1, new Range(ranges[i], ranges[i+1]));
			list.addAll(list2);
		}
		
		return list;
	}

	@Override
	public DataSet<E> test(){
		for(Map<String, Range> combination : combinations)
			for (Instance each : this.testDataSet)
				if(this.getCondition(each, combination))
					each.putResult("result", this.constants.get(combination));
		
		return this.testDataSet;
	}
	
	private class Range{
		double start;
		double end;
		
		public Range(Double start, Double end) {
			this.start = start;
			this.end = end;
		}

		@Override
		public String toString(){
			return "[" + start + "," + end + ")";
		}
	}

}
