package ch.eonum.pipeline.classification.tree;


import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;

/**
 * Random Forest. @link http://en.wikipedia.org/wiki/Random_forest An ensemble
 * of decision trees for classification. Extends the standard Bagging ensemble
 * method by decision tree specific parameters.
 * 
 * @author tim
 * 
 * @param <E>
 */
public class RandomForestClassifier<E extends Instance> extends RandomForest<E>  {

	public RandomForestClassifier(Features features, int seed) {
		super(features, seed);
	}
	
	@Override
	public void train(){
		prepareClasses();
		super.train();
	}
	
	@Override
	protected DecisionTree<E> createDecisionTree(int i) {
		return new DecisionTreeClassifier<E>(features, rand.nextInt(), splitsPerFeature, i, classes);
	}
	
	@Override
	public DataSet<E> test() {
		for(String className : classes.asSet()){
			String label = "likelihoodOfClass" + className;
			for(E e : testDataSet)
				e.removeResult(label);
		}
		for(E e : testDataSet)
			e.removeResult("numTrees");
		
		for(Classifier<E> c : baseClassifiers){
			c.setTestSet(testDataSet);
			testDataSet = c.test();
			for(Instance each : testDataSet)
				if(!c.getTrainingDataSet().contains(each) || !this.oobTesting){
					String label = "likelihoodOfClass" + each.label;
					each.putResult(label, each.getResult(label) + 1.0);
					each.putResult("numTrees", each.getResult("numTrees") + 1.0);
				}
		}
		for(Instance each : testDataSet){
			if(each.getResult("numTrees") != 0){
				double max = Double.NEGATIVE_INFINITY;
				String maxLabel = null;
				for(String className : classes.asSet()){
					String label = "likelihoodOfClass" + className;
					each.putResult(label, each.getResult(label) / each.getResult("numTrees"));
					if(each.get(label) > max){
						max = each.getResult(label);
						maxLabel = className;
					}
				}
				each.label = maxLabel;
				each.putResult("result", max);
			}
		}
		
		for(E e : testDataSet)
			e.removeResult("numTrees");
		
		return this.testDataSet;
	}

}
