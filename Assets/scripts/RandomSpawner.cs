using UnityEngine;

public class RandomSpawner : MonoBehaviour
{
    public GameObject[] prefabs;

    public int count = 100;

    public LayerMask groundLayer;

    public float minScale = 0.8f;
    public float maxScale = 1.5f;

    public float minDistance = 2f; // spacing between objects
    
    public float maxSlopeAngle = 30f; // maximum slope angle (0-90)

    private Vector3[] spawnedPositions;

    [ContextMenu("Spawn Objects")]
    void Spawn()
    {
        // Get the active terrain
        Terrain terrain = Terrain.activeTerrain;
        if (terrain == null)
        {
            Debug.LogError("No active terrain found in the scene!");
            return;
        }

        Vector3 terrainPos = terrain.transform.position;
        Vector3 terrainSize = terrain.terrainData.size;

        spawnedPositions = new Vector3[count];

        int spawned = 0;
        int attempts = 0;

        while (spawned < count && attempts < count * 10)
        {
            attempts++;

            // Random position within terrain bounds
            float randomX = Random.Range(0, terrainSize.x);
            float randomZ = Random.Range(0, terrainSize.z);

            Vector3 pos = new Vector3(
                terrainPos.x + randomX,
                terrainPos.y + 200f,
                terrainPos.z + randomZ
            );

            RaycastHit hit;

            if (Physics.Raycast(pos, Vector3.down, out hit, 200f, groundLayer))
            {
                // Check slope angle
                if (Vector3.Angle(hit.normal, Vector3.up) > maxSlopeAngle)
                    continue;

                Vector3 spawnPoint = hit.point;

                // spacing check
                if (!IsTooClose(spawnPoint, spawned))
                {
                    // Align with terrain slope and random Y rotation
                    Quaternion rotation = Quaternion.FromToRotation(Vector3.up, hit.normal) *
                                          Quaternion.Euler(0, Random.Range(0, 360), 0);
                    
                    GameObject obj = Instantiate(
                        prefabs[Random.Range(0, prefabs.Length)],
                        spawnPoint,
                        rotation,
                        transform
                    );

                    // random scale
                    float scale = Random.Range(minScale, maxScale);
                    obj.transform.localScale = Vector3.one * scale;

                    spawnedPositions[spawned] = spawnPoint;
                    spawned++;
                }
            }
        }
        
        Debug.Log($"Spawned {spawned} objects across terrain after {attempts} attempts");
    }

    bool IsTooClose(Vector3 pos, int count)
    {
        for (int i = 0; i < count; i++)
        {
            if (Vector3.Distance(pos, spawnedPositions[i]) < minDistance)
                return true;
        }
        return false;
    }
    
    // Optional: Visualize terrain bounds in Scene view
    void OnDrawGizmosSelected()
    {
        Terrain terrain = Terrain.activeTerrain;
        if (terrain != null)
        {
            Gizmos.color = Color.green;
            Vector3 terrainSize = terrain.terrainData.size;
            Gizmos.DrawWireCube(terrain.transform.position + terrainSize * 0.5f, terrainSize);
        }
    }
}