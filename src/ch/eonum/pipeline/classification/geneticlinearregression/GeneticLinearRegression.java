package ch.eonum.pipeline.classification.geneticlinearregression;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import Jama.Matrix;
import Jama.QRDecomposition;

import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;

/**
 * Linear Regression model with input vectors created and transformed by a blend
 * of genetic programming and a genetic algorithm.
 * 
 * @author tim
 * 
 */
public class GeneticLinearRegression<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("populationSize", "number of genomes in population (default 100)");
		PARAMETERS.put("maxGenerations", "maximum number of generations (default 100)");
		PARAMETERS.put("maxGenerationsAfterMax", "maximum number of generations after " +
				"the maximum on the validation set is reached (default 20)");
		PARAMETERS.put("maxNodes", "maximum nodes (feature length) for a genome (default 100)");
		PARAMETERS.put("numPopulations", "number of sub populations (default 10)");
		PARAMETERS.put("maxDepth", "maximum tree depth of a node. (default: 3)");
		PARAMETERS.put("slices", "number of slices for the training set. (default: 5)");
	}

	/** genomes. */
	private List<Genome> genomes;
	/** logger. */
	private PrintStream log;
	/** best N genomes from this iteration. */
	private List<Genome> currentBestIndividuals;
	/** best N genomes. */
	private List<Genome> bestIndividuals;
	/** list with the current best genomes. best genome from each population. */
	private ArrayList<Genome> bestIndividualsPerPopulation;
	/** evaluator for fitness function. */
	private Evaluator<E> evaluator;
	private Map<String, Map<Integer, Double>> validationCurves;
	private int exceptionsDuringDecomposition;
	private boolean enableUseOfTestData;

	public GeneticLinearRegression(Features features, Evaluator<E> evaluator) {
		this.setSupportedParameters(GeneticLinearRegression.PARAMETERS);
		this.putParameter("populationSize", 100);
		this.putParameter("maxGenerations", 100);
		this.putParameter("maxGenerationsAfterMax", 20);
		this.putParameter("maxNodes", 100.0);
		this.putParameter("numPopulations", 10);
		this.putParameter("maxDepth", 3.0);
		this.putParameter("slices", 5);
		this.evaluator = evaluator;
		this.features = features;
		this.enableUseOfTestData = false;
	}


	@Override
	public void train() {
		try {
			this.log = new PrintStream(new File(this.baseDir + "geneticLR.txt"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		int slices = this.getIntParameter("slices");
		this.currentBestIndividuals = new ArrayList<Genome>(slices);
		List<DataSet<E>> trainDataSets = this.trainingDataSet.shallowCopy().splitIntoNSubsets(slices);
		
		int populationSize = this.getIntParameter("populationSize");
		
		this.initRandomPopulation(populationSize);
		
		double globalBest = Double.NEGATIVE_INFINITY;
		this.validationCurves = new LinkedHashMap<String, Map<Integer, Double>>();
		validationCurves.put("fitness", new LinkedHashMap<Integer, Double>());
		validationCurves.put("fitness-training", new LinkedHashMap<Integer, Double>());
		validationCurves.put("introns", new LinkedHashMap<Integer, Double>());
		validationCurves.put("genomeSize", new LinkedHashMap<Integer, Double>());
		validationCurves.put("averageDepth", new LinkedHashMap<Integer, Double>());
		validationCurves.put("exceptions", new LinkedHashMap<Integer, Double>());
		
		int generationAfterMax = 0;
		
		for(int generation = 0; generation < this.getIntParameter("maxGenerations"); generation++){
			int slicenumber = generation % slices;
			DataSet<E> fitnessData = trainDataSets.get(slicenumber);
			DataSet<E> lrTrainData = trainDataSets.get((generation + 1) % slices);
			
			double currentBest = this.selectBest(generation, lrTrainData, fitnessData, slicenumber);
			log("Generation " + generation + ": " + currentBest);
			
			
			double evalValidation = this.evaluateList(this.currentBestIndividuals,
					generation, -1, -1,
					trainingDataSet.asDoubleArrayMatrix(features),
					trainingDataSet.outComesAsArray(), this.testDataSet, false);
			log("Generation " + generation + " on the test set: " + evalValidation);
			validationCurves.get("fitness").put(generation, evalValidation);
			
			if(globalBest < evalValidation){
				this.bestIndividuals = this.currentBestIndividuals;
				globalBest = evalValidation;
				generationAfterMax = 0;
			} else
				generationAfterMax++;
			if(generationAfterMax > this.getIntParameter("maxGenerationsAfterMax"))
				break;
			this.statistics(generation);
			for(String curve : validationCurves.keySet())
				Gnuplot.plotOneDimensionalCurve(validationCurves.get(curve),
						curve, this.getBaseDir() + curve);
			this.generateNewPopulation(generation);
		}
		
		log.println("\n== Best individual ==\n" + this.bestIndividuals);
		
		/** train best genome with all data and store beta for testing. */
		DataSet<E> train = this.trainingDataSet;
		if(this.enableUseOfTestData){
			train = train.shallowCopy();
			train.addAll(testDataSet);
		}
		
		this.evaluateList(bestIndividuals, -1, -1, -1,
				train.asDoubleArrayMatrix(features),
				train.outComesAsArray(), testDataSet, true);
		
		this.save(baseDir + "matrixBeta");
		
		log.close();
	}

	private void statistics(int generation) {
		double introns = 0;
		double genomeSize = 0;
		double averageDepth = 0;
		Map<String, Double> nodeTypes = new HashMap<String, Double>();
		Map<String, Double> params = new HashMap<String, Double>();
		for(Genome genome : genomes){
			introns += genome.getNumIntrons();
			genomeSize += genome.size();
			averageDepth += genome.getAverageDepth();
			genome.updateNodeDistribution(nodeTypes);
			genome.updateParameterStatistics(params);
		}
		for(String nodeType : nodeTypes.keySet()){
			if(!validationCurves.containsKey(nodeType))
				validationCurves.put(nodeType, new LinkedHashMap<Integer, Double>());
			validationCurves.get(nodeType).put(generation, nodeTypes.get(nodeType)/genomeSize);
		}
		for(String param : params.keySet()){
			if(!validationCurves.containsKey(param))
				validationCurves.put(param, new LinkedHashMap<Integer, Double>());
			validationCurves.get(param).put(generation, params.get(param)/genomes.size());
		}
		introns /= genomeSize;
		genomeSize /= genomes.size();
		averageDepth /= genomes.size();
		validationCurves.get("introns").put(generation, introns);
		validationCurves.get("genomeSize").put(generation, genomeSize);
		validationCurves.get("averageDepth").put(generation, averageDepth);
		validationCurves.get("exceptions").put(generation, (double)exceptionsDuringDecomposition);
		exceptionsDuringDecomposition = 0;
	}


	public void log(String string) {
		Log.puts(string);
		log.println(string);
	}

	/**
	 * create new population using reproduction, mutation and crossover.
	 */
	private void generateNewPopulation(int generation) {
		Random random = new Random(generation);
		log("Generating new population..");
		this.genomes = new ArrayList<Genome>();
//		for(Genome g : bestIndividuals)
//			g.addIntrons();
		this.genomes.addAll(this.bestIndividualsPerPopulation);
		while(genomes.size() < this.getIntParameter("populationSize")){
			/** choose two random parents. */
			Genome p1 = bestIndividualsPerPopulation.get(random.nextInt(bestIndividualsPerPopulation.size())).copy();
			Genome p2 = bestIndividualsPerPopulation.get(random.nextInt(bestIndividualsPerPopulation.size())).copy();
			
			// mutate
			if(random.nextDouble() < p1.getParameter("mutation")){
				int maxDepth = (int)this.getDoubleParameter("maxDepth");
				Node mutation = Node.randomNode(0.25, 0.25, 0.25, random.nextDouble(),
						features, random, 0, maxDepth == 0 ? 0 : random.nextInt(maxDepth) + 1);
				if(random.nextDouble() < p1.getParameter("replacement") && !(p1.size() < 2))
					p1.replaceNode(random.nextInt(p1.size()), mutation);
				else
					p1.addNode(mutation);
				if(random.nextDouble() < p1.getParameter("removal") && p1.size() > 2)
					p1.removeNode(random.nextInt(p1.size()));
				// mutate parameters
				String param = Genome.paramTypes[random.nextInt(Genome.paramTypes.length)];
				double delta = (random.nextDouble() - 0.5) * 0.3;
				p1.putParameter(param, Math.min(0.9, Math.max(0.1, p1.getParameter(param) + delta)));
			}
			// crossover
			if(random.nextDouble() < p1.getParameter("crossover"))
				p1.crossover(p2);
			
			genomes.add(p1);
		}
		Collections.shuffle(genomes, random);
	}

	/**
	 * fitness function.
	 * @param genome
	 * @param generation
	 * @param population
	 * @param individual
	 * @param trainMatrix
	 * @param testData
	 * @return
	 */
	public double evaluate(Genome genome, int generation, int population,
			int individual, double[][] trainMatrix,
			double[] trainMatrixTargets, DataSet<E> testData, boolean saveBeta) {
		List<Genome> genomes = new ArrayList<Genome>();
		genomes.add(genome);
		return evaluateList(genomes, generation, population, individual,
				trainMatrix, trainMatrixTargets, testData, saveBeta);
	}
	
	public double evaluateList(List<Genome> genomes, int generation, int population,
			int individual, double[][] trainMatrix,
			double[] trainMatrixTargets, DataSet<E> testData, boolean saveBeta) {
		for (Instance each : testData)
			each.put("result", 0.0);

		for (Genome genome : genomes) {
			// train linear regression model
			double[][] trainDataTransformed = genome.transform(trainMatrix);
			int n = genome.size() + 1;
			int m = trainMatrix.length;
			double[][] t = new double[m][1];
			for (int i = 0; i < m; i++)
				t[i][0] = trainMatrixTargets[i];
			Matrix Y = new Matrix(t);
			Matrix X = new Matrix(trainDataTransformed);
			Matrix XtX1 = (X.transpose().times(X));
			Matrix beta = null;
			try {
				XtX1 = new QRDecomposition(XtX1).solve(Matrix.identity(
						XtX1.getRowDimension(), XtX1.getRowDimension()));
				beta = XtX1.times(X.transpose()).times(Y);
			} catch (Exception e) {
				this.exceptionsDuringDecomposition++;
				beta = new Matrix(n, 1);
			}

			if (saveBeta) {
				genome.setBeta(beta);
			}
			// test on test data
			double[][] testDataTransformed = genome.transform(testData
					.asDoubleArrayMatrix(features));
			for (int i = 0; i < testDataTransformed.length; i++) {
				double prediction = 0.0;
				for (int x = 0; x < n; x++)
					prediction += testDataTransformed[i][x] * beta.get(x, 0);
				testData.get(i).putResult("result",
						testData.get(i).getResult("result") + prediction);
			}
		}

		for (Instance each : testData)
			each.putResult("result", each.getResult("result") / genomes.size());

		return evaluator.evaluate(testData);
	}

	/**
	 * select the best individuals, one from each sub population.
	 * => tournament selection
	 * @param generation
	 * @param trainData
	 * @param fitnessData
	 * @param slicenumber 
	 * @return
	 */
	private double selectBest(int generation, DataSet<E> trainData, DataSet<E> fitnessData, int slicenumber) {
		int numPopulations = this.getIntParameter("numPopulations");
		this.bestIndividualsPerPopulation = new ArrayList<Genome>(numPopulations);

		// split population into sub populations
		List<List<Genome>> subPopulations = new ArrayList<List<Genome>>();
		for (int population = 0; population < numPopulations; population++)
			subPopulations.add(new ArrayList<Genome>());
		for (int i = 0; i < genomes.size(); i++)
			subPopulations.get(i % numPopulations).add(genomes.get(i));

		/** get the best individual from each sub population. */
		ExecutorService service = Executors.newFixedThreadPool(Math.min(Runtime
				.getRuntime().availableProcessors(), numPopulations));
		List<SubpopulationEvaluator<E>> populationEvaluators = new ArrayList<SubpopulationEvaluator<E>>(numPopulations);
		for (int population = 0; population < numPopulations; population++) {
			SubpopulationEvaluator<E> se = new SubpopulationEvaluator<E>(
					population, generation, subPopulations.get(population),
					this, trainData, fitnessData);
			populationEvaluators.add(se );
			service.submit(se);
		}

		 service.shutdown();
		 try {
			 service.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		 } catch (InterruptedException e) {
			 e.printStackTrace();
			 System.exit(-1);
		 }

		for (int population = 0; population < numPopulations; population++)
			this.bestIndividualsPerPopulation.add(populationEvaluators.get(population).getBestGenome());
		
		Collections.sort(this.bestIndividualsPerPopulation);
		if(slicenumber >= this.currentBestIndividuals.size())
			this.currentBestIndividuals.add(bestIndividualsPerPopulation.get(0));
		else
			this.currentBestIndividuals.set(slicenumber, bestIndividualsPerPopulation.get(0));

		double best = this.bestIndividualsPerPopulation.get(0).getFitness();
		validationCurves.get("fitness-training").put(generation, best);
		return best;
	}


	private void initRandomPopulation(int populationSize) {
		int maxNodes = (int)this.getDoubleParameter("maxNodes");
		int maxDepth = (int)this.getDoubleParameter("maxDepth");
		this.genomes = new ArrayList<Genome>();
		for(int i = 0; i < populationSize; i++)
			genomes.add(new Genome(features, i, maxNodes, maxDepth));
	}
	
	@Override
	public DataSet<E> test(){
		for(Instance each : testDataSet)
			each.put("result", 0.0);
		for(Genome genome : this.bestIndividuals){
			double[][] testDataTransformed = genome.transform(testDataSet
					.asDoubleArrayMatrix(features));
			for(int i = 0; i < testDataTransformed.length; i++){
				double prediction = 0.0;
				for(int x = 0; x < genome.getBeta().getRowDimension(); x++)
					prediction += testDataTransformed[i][x] * genome.getBeta().get(x, 0);
				testDataSet.get(i).putResult(
						"result",
						testDataSet.get(i).getResult("result")
								+ Math.max(0, prediction));
			}
		}
		
		for(Instance each : testDataSet)
			each.putResult("result", each.getResult("result")/bestIndividuals.size());

		return this.testDataSet;
	}
	
	@Override
	public void save(String fileName) {
		try {
			PrintStream p = new PrintStream(new FileOutputStream(fileName));
			for(Genome genome : this.bestIndividuals){
				for(int x = 0; x < genome.getBeta().getColumnDimension(); x++){
					for(int y = 0; y < genome.getBeta().getRowDimension(); y++)
						p.print(genome.getBeta().get(y,x) + " ");
					p.println();
				}
			}
			p.println();
			p.println(bestIndividuals);
			p.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * enable the use of the test set for the final training
	 */
	public void enableUseOfTestData() {
		this.enableUseOfTestData = true;
	}

}
