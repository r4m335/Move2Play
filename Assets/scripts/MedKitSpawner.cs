using UnityEngine;
using System.Collections.Generic;

public class MedKitSpawner : MonoBehaviour
{
    public GameObject medKitPrefab;

    public int medKitCount = 10;

    public LayerMask groundLayer;
    public LayerMask waterLayer;

    public float minDistanceBetweenMedkits = 15f;
    public float maxSlopeAngle = 30f;

    // Track all spawned positions to avoid clustering
    private List<Vector3> spawnedPositions = new List<Vector3>();

    void Start()
    {
        SpawnMedkits();
    }

    bool IsTooClose(Vector3 pos)
    {
        foreach (Vector3 existingPos in spawnedPositions)
        {
            if (Vector3.Distance(pos, existingPos) < minDistanceBetweenMedkits)
                return true;
        }
        return false;
    }

    void SpawnMedkits()
    {
        Terrain terrain = Terrain.activeTerrain;

        if (terrain == null)
            return;

        Vector3 terrainPos = terrain.transform.position;
        Vector3 terrainSize = terrain.terrainData.size;

        int spawned = 0;
        int attempts = 0;

        while (spawned < medKitCount && attempts < medKitCount * 20)
        {
            attempts++;

            float rx = Random.Range(0, terrainSize.x);
            float rz = Random.Range(0, terrainSize.z);

            Vector3 rayStart =
                new Vector3(
                    terrainPos.x + rx,
                    terrainPos.y + 200f,
                    terrainPos.z + rz);

            if (Physics.Raycast(rayStart,
                Vector3.down,
                out RaycastHit hit,
                300f,
                groundLayer))
            {
                // Slope check
                if (Vector3.Angle(hit.normal, Vector3.up) > maxSlopeAngle)
                    continue;

                // Water check
                if (((1 << hit.collider.gameObject.layer) & waterLayer) != 0)
                    continue;

                // New distance check: skip if too close to any already spawned medkit
                if (IsTooClose(hit.point))
                    continue;

                Instantiate(
                    medKitPrefab,
                    hit.point + Vector3.up * 0.5f,
                    Quaternion.identity,
                    transform);

                spawnedPositions.Add(hit.point);
                spawned++;
            }
        }

        Debug.Log($"Spawned {spawned} medkits");
    }
}