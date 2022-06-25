using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Azure.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json.Linq;

namespace functions
{
    public static class GetModels
    {
        [FunctionName("GetModels")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            var creds = new DefaultAzureCredential();
            var accessToken = await creds.GetTokenAsync(new Azure.Core.TokenRequestContext(new[] { "https://management.azure.com/.default" }));

            var rg = Environment.GetEnvironmentVariable("ResourceGroup");
            var subscriptionId = Environment.GetEnvironmentVariable("SubscriptionId");
            var workspace = Environment.GetEnvironmentVariable("MLWorkspace");
            string url = $"https://management.azure.com/subscriptions/{subscriptionId}/resourceGroups/{rg}/providers/Microsoft.MachineLearningServices/workspaces/{workspace}/models?api-version=2022-05-01&$skip=0&count=10";
            HttpClient client = new HttpClient();
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", accessToken.Token);

            HttpResponseMessage response = await client.GetAsync(url);
            response.EnsureSuccessStatusCode();
            string responseBody = await response.Content.ReadAsStringAsync();

            var data = JObject.Parse(responseBody);

            JToken valueToken = null;
            data.TryGetValue("value", out valueToken);

            var sb = new StringBuilder("{\"models\":[");
            foreach (var x in valueToken)
            {
                sb.Append("{");
                sb.AppendFormat("\"name\":\"{0}\"", x["name"]);
                sb.Append("},");
            }

            if (sb.Length > 1)
            {
                sb.Remove(sb.Length-1, 1);
            }

            sb.Append("]}");

            log.LogInformation("Response: " + sb.ToString());

            return new JsonResult(JObject.Parse(sb.ToString()));
        }
    }
}

