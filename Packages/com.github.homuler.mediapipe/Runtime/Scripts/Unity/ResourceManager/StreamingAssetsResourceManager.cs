// Copyright (c) 2021 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace Mediapipe.Unity
{
  public class StreamingAssetsResourceManager : IResourceManager
  {
    private static readonly string _TAG = nameof(StreamingAssetsResourceManager);

    private static string _RelativePath;
    private static string _AssetPathRoot;
    private static string _CachePathRoot;

    public StreamingAssetsResourceManager(string path)
    {
      ResourceUtil.EnableCustomResolver();
      _RelativePath = path;
      _AssetPathRoot = Path.Combine(Application.streamingAssetsPath, _RelativePath);
      _CachePathRoot = Path.Combine(Application.persistentDataPath, _RelativePath);
    }

    public StreamingAssetsResourceManager() : this("") { }

    IEnumerator IResourceManager.PrepareAssetAsync(string name, string uniqueKey, bool overwriteDestination)
    {
      var destFilePath = GetCachePathFor(uniqueKey);
      ResourceUtil.SetAssetPath(name, destFilePath);

      if (File.Exists(destFilePath) && !overwriteDestination)
      {
        Logger.LogInfo(_TAG, $"{name} will not be copied to {destFilePath} because it already exists");
        yield break;
      }

      var sourceFilePath = GetCachePathFor(name);
      if (!File.Exists(sourceFilePath))
      {
        yield return CreateCacheFile(name);
      }

      if (sourceFilePath == destFilePath)
      {
        yield break;
      }

      Logger.LogVerbose(_TAG, $"Copying {sourceFilePath} to {destFilePath}...");
      File.Copy(sourceFilePath, destFilePath, overwriteDestination);
      Logger.LogVerbose(_TAG, $"{sourceFilePath} is copied to {destFilePath}");
    }

    private IEnumerator CreateCacheFile(string assetName)
    {
      var cacheFilePath = GetCachePathFor(assetName);
      var assetPath = GetAssetPathFor(assetName);

      Debug.Log("StreamingAssets Path : " + Application.streamingAssetsPath);
      Debug.Log("Persistent Path      : " + Application.persistentDataPath);
      Debug.Log("Asset Path           : " + assetPath);
      Debug.Log("Cache Path           : " + cacheFilePath);
      Debug.Log("Asset Exists         : " + File.Exists(assetPath));
      Debug.Log("Cache Exists         : " + File.Exists(cacheFilePath));

      if (File.Exists(cacheFilePath))
      {
        yield break;
      }

      if (!File.Exists(assetPath))
      {
        throw new FileNotFoundException($"StreamingAssets file not found: {assetPath}");
      }

      if (!Directory.Exists(_CachePathRoot))
      {
        Directory.CreateDirectory(_CachePathRoot);
      }

      Logger.LogVerbose(_TAG, $"Copying {assetPath} to {cacheFilePath}...");

      File.Copy(assetPath, cacheFilePath, true);

      Logger.LogVerbose(_TAG, $"{assetName} copied successfully.");
    }

    private static string GetAssetPathFor(string assetName)
    {
      return Path.Combine(_AssetPathRoot, assetName);
    }

    private static string GetCachePathFor(string assetName)
    {
      var assetPath = GetAssetPathFor(assetName);
      return File.Exists(assetPath) ? assetPath : Path.Combine(_CachePathRoot, assetName);
    }
  }
}