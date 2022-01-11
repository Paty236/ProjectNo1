
using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Recommender;
using System.Reflection;
using System.Reflection.Emit;
using System.Data;

public class MovieRating
{
    [LoadColumn(0)] public float userId;
    [LoadColumn(1)] public float movieId;
    [LoadColumn(2)] public float Label;    
}
public class MovieRatingPrediction
{
    public float Label;
    public float Score;
}
public class Movie
{
    public int ID;
    public String Title;
}
public static class Movies
{
    public static List<Movie> All = new List<Movie>();
   
    public static Movie Get(int id)
    {
        return All.Single(m => m.ID == id);
    }
}
public static class CsvToDataTable
{
    public static DataTable ConvertCsvToDataTable(string filePath)
    {
        //reading all the lines(rows) from the file.
        string[] rows = File.ReadAllLines(filePath);

        DataTable dtData = new DataTable();
        string[] rowValues = null;
        DataRow dr = dtData.NewRow();

        //Creating columns
        if (rows.Length > 0)
        {
            foreach (string columnName in rows[0].Split(','))
                dtData.Columns.Add(columnName);
        }

        //Creating row for each line.(except the first line, which contain column names)
        for (int row = 1; row < rows.Length; row++)
        {
            rowValues = rows[row].Split(',');
            dr = dtData.NewRow();
            dr.ItemArray = rowValues;
            dtData.Rows.Add(dr);
        }

        return dtData;
    }
}
class Program
{
    private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "recommendation-ratings-test.csv");
    private static string moviesDataPath = Path.Combine(Environment.CurrentDirectory, "recommendation-movies.csv");

    static void Main(string[] args)
    {
        var mlContext = new MLContext();
        var testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

        // prepare matrix factorization options
        var options = new MatrixFactorizationTrainer.Options
        {
            MatrixColumnIndexColumnName = "userIdEncoded",
            MatrixRowIndexColumnName = "movieIdEncoded",
            LabelColumnName = "Label",
            NumberOfIterations = 20,
            ApproximationRank = 100
        };

        // set up a training pipeline
        // step 1: map userId and movieId to keys
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "userId",
                outputColumnName: "userIdEncoded")
            .Append(mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "movieId",
                outputColumnName: "movieIdEncoded")

          // step 2: find recommendations using matrix factorization
            .Append(mlContext.Recommendation().Trainers.MatrixFactorization(options)));

        // train the model
        var model = pipeline.Fit(testDataView);
        Console.WriteLine();

        var predictions = model.Transform(testDataView);
        var metrics = mlContext.Regression.Evaluate(predictions);
        Console.WriteLine();


        Console.WriteLine("Select user id (id from 1 - 50): ");
        string inptVal = Console.ReadLine();
        int userId = 1;
        if(int.TryParse(inptVal, out userId) == false) Console.WriteLine("Value is not number");
        Console.WriteLine(String.Format("Calculating the score for user {0} liking the movie 'GoldenEye'...", userId.ToString()));
        var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
        var prediction = predictionEngine.Predict(
            new MovieRating()
            {
                userId = userId,
                movieId = 10  // GoldenEye
            }
        );
        Console.WriteLine($"  Score: {prediction.Score}");
        Console.WriteLine();

        // find the top 5 movies for a given user

        DataTable movies = CsvToDataTable.ConvertCsvToDataTable(moviesDataPath);
        Movies.All = movies.AsEnumerable().Select(r => new Movie { ID = Convert.ToInt32(r[0]), Title = r[1].ToString() }).ToList();

        Console.WriteLine(String.Format("Calculating the top 5 movies for user {0}...", userId.ToString()));
        var top5 = (from m in Movies.All
                    let p = predictionEngine.Predict(
                       new MovieRating()
                       {
                           userId = userId,
                           movieId = m.ID
                       })
                    orderby p.Score descending
                    select (MovieId: m.ID, Score: p.Score)).Take(5);

        foreach (var t in top5)
            Console.WriteLine($"  Score:{t.Score}\tMovie: {Movies.Get(t.MovieId)?.Title}");

        Console.ReadKey();
    }
}
