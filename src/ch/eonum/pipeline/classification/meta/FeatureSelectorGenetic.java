package ch.eonum.pipeline.classification.meta;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.util.Log;

/**
 * Genetic feature selection algorithm.
 * A genome is a list with a flag for each feature whether to use it or not.
 * 
 * @author tim
 *
 * @param <E>
 */
public class FeatureSelectorGenetic<E extends Instance> extends Classifier<E> {

protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("maxIterations", "maximum number of iterations. (default: 100)");
		PARAMETERS.put("populations", "number of subpopulations. (default: 10.0)");
		PARAMETERS.put("initialPopulationSize", "size of the initial population. (default: 100.0)");
		PARAMETERS.put("crossover", "crossover probability. (default: 0.6)");
		PARAMETERS.put("mutation", "mutation probability. (default: 0.001)");
	}

	private Classifier<E> baseClassifier;
	private Features reducedFeatures;
	private Evaluator<E> evaluator;
	private int featureSize;
	private boolean[][] genes;
	private Features currentBestFeatures;
	private PrintStream printer;
	private Random random;

	public FeatureSelectorGenetic(Classifier<E> baseClassifier, Features features, Evaluator<E> evaluator) {
		this.evaluator = evaluator;
		this.baseClassifier = baseClassifier;
		this.features = features;
		this.testDataSet = this.baseClassifier.getTestDataSet();
		this.trainingDataSet = this.baseClassifier.getTrainingDataSet();
		this.setSupportedParameters(FeatureSelectorGenetic.PARAMETERS);
		this.putParameter("maxIterations", 100);
		this.putParameter("populations", 10.0);
		this.putParameter("initialPopulationSize", 100.0);
		this.putParameter("crossover", 0.6);
		this.putParameter("mutation", 0.001);
		this.random = new Random(23);
	}

	@Override
	public void train() {
		try {
			this.printer = new PrintStream(new File(this.baseDir + "feature-selection.txt"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		this.baseClassifier.setTrainingSet(trainingDataSet);
		this.baseClassifier.setTestSet(testDataSet);
		this.featureSize = this.features.size();
		this.generateInitialPopulation();
		
		double globalBest = Double.NEGATIVE_INFINITY;
		
		for(int iteration = 0; iteration < this.getIntParameter("maxIterations"); iteration++){
			double currentBest = this.selectBest();
			if(globalBest < currentBest){
				this.reducedFeatures = this.currentBestFeatures;
				globalBest = currentBest;
			}
			Log.puts("Iteration " + iteration + ": " + currentBest);
			printer.println("Iteration " + iteration + ": " + currentBest);
			this.generateNewPopulation();
			reducedFeatures.writeToFile(baseDir + "reduced-features-genetic.txt");
		}
		
		baseClassifier.setFeatures(reducedFeatures);
		baseClassifier.train();
		printer.close();
	} 

	private double selectBest() {
		double best = Double.NEGATIVE_INFINITY;
		/** divide population. */
		int numPopulations  = (int)this.getDoubleParameter("populations");
		boolean populations[][][] = new boolean[numPopulations][genes.length/numPopulations][];
		for(int i = 0; i < genes.length; i++)
			populations[i%numPopulations][(i - (i%numPopulations))/numPopulations] = genes[i];
		this.genes = new boolean[numPopulations][];
		/** get best from each population. */
		for(int population = 0; population < numPopulations; population++){
			double max = Double.NEGATIVE_INFINITY;
			int maxI = -1;
			for(int i = 0; i < populations.length; i++){
				double eval = this.evaluateFeatures(populations[population][i]);
				if(eval > max){
					max = eval;
					maxI = i;
				}
			}
			Log.puts("Best gene from population " + population + ": " + max);
			printer.println("Best gene from population " + population + ": " + max);
			this.genes[population] = populations[population][maxI];
			if(best < max){
				best = max;
				this.currentBestFeatures = this.getFeaturesFromGene(populations[population][maxI]);
			}
		}
		return best;
	}

	/**
	 * Fitness function
	 * @param f
	 * @return
	 */
	private double evaluateFeatures(boolean[] f) {
		Features fea = this.getFeaturesFromGene(f);
		this.baseClassifier.setFeatures(fea);
		this.baseClassifier.setTrainingSet(trainingDataSet);
		this.baseClassifier.setTestSet(testDataSet);
		this.baseClassifier.train();		
		return this.evaluator.evaluate(this.baseClassifier.test());
	}

	private Features getFeaturesFromGene(boolean[] f) {
		Features fea = new Features();
		for(int i = 0; i < featureSize; i++)
			if(f[i])
				fea.addFeature(features.getFeatureByIndex(i));
		return fea;
	}

	private void generateInitialPopulation() {
		this.genes = new boolean[(int)this.getDoubleParameter("initialPopulationSize")][this.featureSize];
		double threshold = this.random.nextDouble() * 0.9 + 0.1;
		for(int g = 0; g < genes.length; g++)
			for(int f = 0; f < this.featureSize; f++)
				genes[g][f] = this.random.nextDouble() > threshold;
	}

	private void generateNewPopulation() {
		Log.puts("Generating new population..");
		double crossover = this.getDoubleParameter("crossover");
		double mutation = this.getDoubleParameter("mutation");
		boolean[][] parents = this.genes;
		this.genes = new boolean[(int)this.getDoubleParameter("initialPopulationSize")][];
		int added = 0;
		for(boolean[] each : parents){
			genes[added] = each;
			added++;
		}
		while(added < genes.length){
			boolean[] gene1 = new boolean[featureSize];
			boolean[] parent = parents[(int)(this.random.nextDouble() * parents.length)];
			for(int f = 0; f < featureSize; f++)
				gene1[f] = parent[f];
			boolean[] gene2 = new boolean[featureSize];
			parent = parents[(int)(this.random.nextDouble() * parents.length)];
			for(int f = 0; f < featureSize; f++)
				gene2[f] = parent[f];
			
			if(this.random.nextDouble() < crossover){
				for(int f = 0; f < featureSize; f++)
					if(this.random.nextDouble() < 0.5){
						boolean temp = gene1[f];
						gene1[f] = gene2[f];
						gene2[f] = temp;
					}
			}
			if(this.random.nextDouble() < mutation){
				int f = (int)(this.random.nextDouble() * gene1.length);
				gene1[f] = !gene1[f];
				gene2[f] = !gene2[f];
			}
			genes[added] = gene1;
			added++;
			if(added < genes.length){
				genes[added] = gene2;
				added++;
			}
		}
		/** shuffle. */
		for(int i = 0; i < genes.length; i++){
			int x = (int)(this.random.nextDouble() * genes.length);
			int y = (int)(this.random.nextDouble() * genes.length);
			boolean[] temp = genes[x];
			genes[x] = genes[y];
			genes[y] = temp;
		}
	}
	
	@Override
	public DataSet<E> test() {
		baseClassifier.setTestSet(testDataSet);
		baseClassifier.setFeatures(reducedFeatures);
		return baseClassifier.test();
	}

}
