package ch.eonum.pipeline.classification.lstm;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Sequence;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;


/**
 * LSTM refined with a genetic algorithm.
 * 
 * <p>Train some LSTMs and use them as an initial population. Use the weight
 * matrices as genomes. Tournament selection algorithm. The number of
 * sub populations is equal the number of concurrently trained nets, which have
 * been differently initialized.<p>
 * 
 * @author tim
 * 
 */
public class GeneticLSTM<E extends Sequence> extends LSTM<E> {
	
	static {
		PARAMETERS.put("maxIterations", "maximum number of iterations. (default: 100)");
		PARAMETERS.put("maxGenerationsAfterMax", "maximum number of iterations/generations after " +
				"having reached the maximum on the validation set (default: 50)");
		PARAMETERS.put("initialPopulationSize", "size of the initial population. (default: 100.0)");
		PARAMETERS.put("crossover", "crossover probability. (default: 0.6)");
		PARAMETERS.put("mutation", "mutation probability. (default: 0.1)");
	}
	
	private Evaluator<E> fitnessFunction;
	/** pseudo random number generator. */
	private Random random;
	/** genes represented by weight matrices per individual per population. */
	private double[][][][] genes;
	private double[][] bestMatrix;
	private double[][] currentBestMatrix;
	private PrintStream printer;
	
	public GeneticLSTM(Evaluator<E> fitnessFunction){
		super();
		this.fitnessFunction = fitnessFunction;
		this.putParameter("maxIterations", 100);
		this.putParameter("initialPopulationSize", 100.0);
		this.putParameter("crossover", 1.0);
		this.putParameter("mutation", 0.1);
		this.putParameter("maxGenerationsAfterMax", 50);
		this.random = new Random(2);
	}
	@Override
	protected void refineNets(int maxNet) {
		int oldMaxEpochsAfterMax = this.getIntParameter("maxEpochsAfterMax");
		this.putParameter("maxEpochsAfterMax", 5);
		try {
			this.printer = new PrintStream(new File(this.baseDir + "geneticLSTM.txt"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		// generate initial population
		int numPopulations  = (int)this.getDoubleParameter("numNets");
		int individualPerPopulation = (int)this.getDoubleParameter("initialPopulationSize")/numPopulations;
		this.genes = new double[numPopulations][individualPerPopulation][][];
		for(int iPop = 0; iPop < numPopulations; iPop++){
			FileUtil.mkdir(this.getBaseDir() + "genetic" + iPop + "/");
			for(int iInd = 0; iInd < individualPerPopulation; iInd++)
				this.genes[iPop][iInd] = this.nets.get(iPop).getWeightMatrix();
		}
		
		double globalBest = Double.NEGATIVE_INFINITY;
		Map<Integer, Double> validationCurve = new LinkedHashMap<Integer, Double>();
		
		int iterationsAfterMax = 0;
		
		for(int iteration = 0; iteration < this.getIntParameter("maxIterations"); iteration++){
			double currentBest = this.selectBest(iteration);
			Log.puts("Iteration " + iteration + ": " + currentBest);
			printer.println("Iteration " + iteration + ": " + currentBest);
			
			double evalValidation = this.evaluate(this.currentBestMatrix, iteration,
					-1, -1, this.testDataSet, inputTest, targetTest);
			String log = "Iteration " + iteration + " on the validation set: " + evalValidation;
			Log.puts(log);
			printer.println(log);
			validationCurve.put(iteration, evalValidation);
			Gnuplot.plotOneDimensionalCurve(validationCurve, "Validation Genetic Algorithm on LSTM", this.getBaseDir() + "validation-genetic");
			if(globalBest < evalValidation){
				this.bestMatrix = this.currentBestMatrix;
				globalBest = evalValidation;
				iterationsAfterMax = 0;
			} else
				iterationsAfterMax++;
			if(iterationsAfterMax > this.getIntParameter("maxGenerationsAfterMax"))
				break;
			
			this.generateNewPopulation();
		}
		
		this.nets.get(maxNet).setWeightMatrix(bestMatrix);
		printer.close();
		this.putParameter("maxEpochsAfterMax", oldMaxEpochsAfterMax);
	}
	
	private void generateNewPopulation() {
		Log.puts("Generating new population..");
		double crossover = this.getDoubleParameter("crossover");
		double mutation = this.getDoubleParameter("mutation");
		int matrixLength = this.genes[0][0].length;
		int numPopulations  = (int)this.getDoubleParameter("numNets");
		int individualPerPopulation = (int)this.getDoubleParameter("initialPopulationSize")/numPopulations;
		
		double[][][][] parents = this.genes;
		this.genes = new double[numPopulations][individualPerPopulation][][];
				
		for(int population = 0; population < numPopulations; population++)
			for(int individual = 0; individual < individualPerPopulation; individual++){
				double[][] gene1 = new double[matrixLength][matrixLength];
				double[][] parent = parents[random.nextInt(numPopulations)][0];
				// copy
				for(int x = 0; x < matrixLength; x++)
					for(int y = 0; y < matrixLength; y++)
						gene1[x][y] = parent[x][y];
				double[][] gene2 = new double[matrixLength][matrixLength];
				parent = parents[random.nextInt(numPopulations)][0];
				// copy
				for(int x = 0; x < matrixLength; x++)
					for(int y = 0; y < matrixLength; y++)
						gene2[x][y] = parent[x][y];
				// crossover
				double threshold = random.nextDouble();
				if(random.nextDouble() < crossover){
					int numberOfCrossoverPoints = (int) random
							.nextInt((int) Math.max(1.0, (threshold * (gene1.length / 2.0))));
					for(int i = 0; i < numberOfCrossoverPoints; i++){
						int crossoverpoint = random.nextInt(gene1.length - 1);
						for(int x = 0; x < matrixLength; x++){
							double temp = gene1[x][crossoverpoint];
							gene1[x][crossoverpoint] = gene2[x][crossoverpoint];
							gene2[x][crossoverpoint] = temp;
						}
					}
				}
				// mutation
				if(random.nextDouble() < mutation){
					int x = random.nextInt(gene1.length);
					int y = random.nextInt(gene1.length);
					gene1[x][y] = gene1[x][y] * (random.nextDouble() * 4.0 - 2.0);
				}
				genes[population][individual] = gene1;
			}
		// add the best performing individuals unchanged
		for(double[][][] each : parents){
			genes[this.random.nextInt(numPopulations)][this.random.nextInt(individualPerPopulation)] = each[0];
		}
	}
	
	/**
	 * select the best individual from each population and put it at the first position of each population.
	 * @param iteration
	 * @return
	 */
	private double selectBest(int iteration) {
		
		int numPopulations  = (int)this.getDoubleParameter("numNets");
		List<SubpopulationEvaluator<E>> populationEvaluators = new ArrayList<SubpopulationEvaluator<E>>(numPopulations);
		
		// get best from each population
		ExecutorService service = Executors.newFixedThreadPool(numPopulations);
		for(int population = 0; population < numPopulations; population++){
			SubpopulationEvaluator<E> pe = new SubpopulationEvaluator<E>(
					population, iteration, this.genes[population], this, input,
					target, inputTest, targetTest, outputGates, forgetGates, inputGates);
			populationEvaluators.add(pe );
			service.submit(pe);
		}
		
		service.shutdown();
		try {
			service.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		double best = Double.NEGATIVE_INFINITY;
		
		for(int population = 0; population < numPopulations; population++){
			double max = populationEvaluators.get(population).getMaxIndividualPerformance();
			if(best < max){
				best = max;
				this.currentBestMatrix = genes[population][0];
			}
		}
		
		return best;
	}
	
	public double evaluate(double[][] weightMatrix, int iteration,
			int population, int individual, DataSet<E> dataSet,
			double[][][] inputs, double[][][] targets) {
		for(Sequence each : dataSet)
			each.putResult("result", 0.0);
		
		LSTMCore<E> net = new LSTMCore<E>("Iteration" + iteration + " Population: "
				+ population + " Individual: " + individual, this.getBaseDir(),
				this, 0, outputGates, forgetGates, inputGates, false);
		net.setTestData(inputs, targets);
		net.setWeightMatrix(weightMatrix);
		net.test(null, dataSet);
		return this.fitnessFunction.evaluate(dataSet);
	}
	
	public synchronized void println(String string) {
		this.printer.println(string);
	}

}
