package com.argus.agi.data.local

import android.content.Context
import androidx.room.Room
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ArgusDatabaseMigrationTest {

    @Test
    fun concurrentAccessReturnsOneRoomSingleton() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val executor = Executors.newFixedThreadPool(8)
        val startGate = CountDownLatch(1)

        try {
            val results = (0 until 8).map {
                executor.submit<ArgusDatabase> {
                    check(startGate.await(5, TimeUnit.SECONDS))
                    ArgusDatabase.getInstance(context)
                }
            }

            startGate.countDown()
            val databases = results.map { it.get(10, TimeUnit.SECONDS) }
            val first = databases.first()
            databases.forEach { database ->
                assertSame(first, database)
            }
        } finally {
            startGate.countDown()
            executor.shutdownNow()
            executor.awaitTermination(5, TimeUnit.SECONDS)
        }
    }

    @Test
    fun versionOneUserDataSurvivesAdditiveCorpusMigration() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val databaseName = "argus_migration_v1_test.db"
        context.deleteDatabase(databaseName)

        context.openOrCreateDatabase(databaseName, Context.MODE_PRIVATE, null).use { legacy ->
            legacy.execSQL(
                """
                CREATE TABLE IF NOT EXISTS `chat_messages` (
                    `id` TEXT NOT NULL,
                    `isUser` INTEGER NOT NULL,
                    `text` TEXT NOT NULL,
                    `thinkReasoning` TEXT,
                    `attachedFileName` TEXT,
                    `attachedImageUri` TEXT,
                    `timestamp` INTEGER NOT NULL,
                    PRIMARY KEY(`id`)
                )
                """.trimIndent()
            )
            legacy.execSQL(
                """
                CREATE TABLE IF NOT EXISTS `world_memories` (
                    `key` TEXT NOT NULL,
                    `value` TEXT NOT NULL,
                    `source` TEXT NOT NULL,
                    `timestamp` INTEGER NOT NULL,
                    PRIMARY KEY(`key`)
                )
                """.trimIndent()
            )
            legacy.execSQL(
                """
                CREATE TABLE IF NOT EXISTS `training_items` (
                    `id` TEXT NOT NULL,
                    `source` TEXT NOT NULL,
                    `title` TEXT NOT NULL,
                    `content` TEXT NOT NULL,
                    `chunkIndex` INTEGER NOT NULL,
                    `totalChunks` INTEGER NOT NULL,
                    `documentTitle` TEXT,
                    `timestamp` INTEGER NOT NULL,
                    `status` TEXT NOT NULL,
                    `finalLoss` REAL,
                    `stepDurationMs` INTEGER NOT NULL,
                    PRIMARY KEY(`id`)
                )
                """.trimIndent()
            )
            legacy.execSQL(
                "INSERT INTO `chat_messages` " +
                    "(`id`, `isUser`, `text`, `timestamp`) VALUES ('legacy-chat', 1, 'Keep me', 10)"
            )
            legacy.execSQL(
                "INSERT INTO `world_memories` " +
                    "(`key`, `value`, `source`, `timestamp`) " +
                    "VALUES ('legacy-memory', 'Keep this memory', 'test', 20)"
            )
            legacy.execSQL(
                "INSERT INTO `training_items` " +
                    "(`id`, `source`, `title`, `content`, `chunkIndex`, " +
                    "`totalChunks`, `timestamp`, `status`, `stepDurationMs`) " +
                    "VALUES ('legacy-training', 'test', 'Legacy', 'Keep this queued text', 1, 1, 30, 'QUEUED', 0)"
            )
            legacy.version = 1
        }

        val migrationWithRetainedCorpus = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                ArgusDatabase.MIGRATION_1_2.migrate(db)

                db.execSQL(
                    """
                    INSERT INTO `corpus_documents` (
                        `id`, `contentFingerprint`, `fingerprintAlgorithm`, `source`,
                        `title`, `language`, `canonicalUrl`, `characterCount`,
                        `chunkCount`, `datasetSplit`, `validationText`, `plannedEpochs`,
                        `rightsStatus`, `status`, `createdAtEpochMs`,
                        `lastTrainedAtEpochMs`, `completedPassCount`
                    ) VALUES (
                        'legacy-doc', 'legacy-fingerprint', 'SHA256_UTF16_BE', 'legacy-import',
                        'Legacy article', 'en', 'https://example.org/legacy', 20,
                        1, 'TRAIN', NULL, 1, 'USER_DECLARED', 'PENDING', 200, NULL, 0
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    """
                    INSERT INTO `corpus_sources` (
                        `id`, `documentId`, `source`, `canonicalUrl`,
                        `language`, `rightsStatus`, `importedAtEpochMs`
                    ) VALUES (
                        'legacy-source', 'legacy-doc',
                        'web_article|rights_declared_by_user=true',
                        'https://example.org/legacy', 'en', 'USER_DECLARED', 300
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    """
                    INSERT INTO `corpus_chunks` (
                        `id`, `documentId`, `chunkIndex`, `content`, `characterCount`
                    ) VALUES (
                        'legacy-chunk', 'legacy-doc', 1, 'legacy chunk content', 20
                    )
                    """.trimIndent()
                )

                db.execSQL(
                    """
                    INSERT INTO `corpus_documents` (
                        `id`, `contentFingerprint`, `fingerprintAlgorithm`, `source`,
                        `title`, `language`, `canonicalUrl`, `characterCount`,
                        `chunkCount`, `datasetSplit`, `validationText`, `plannedEpochs`,
                        `rightsStatus`, `status`, `createdAtEpochMs`,
                        `lastTrainedAtEpochMs`, `completedPassCount`
                    ) VALUES (
                        'unverified-doc', 'unverified-fingerprint', 'SHA256_UTF16_BE',
                        'legacy-import', 'Unverified legacy source', 'en', NULL, 20,
                        1, 'TRAIN', NULL, 1, 'LICENSE_REVIEWED', 'PENDING', 400, NULL, 0
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    """
                    INSERT INTO `corpus_sources` (
                        `id`, `documentId`, `source`, `canonicalUrl`,
                        `language`, `rightsStatus`, `importedAtEpochMs`
                    ) VALUES (
                        'unverified-source', 'unverified-doc', 'legacy-license-source',
                        NULL, 'en', 'LICENSE_REVIEWED', 500
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    """
                    INSERT INTO `corpus_chunks` (
                        `id`, `documentId`, `chunkIndex`, `content`, `characterCount`
                    ) VALUES (
                        'unverified-chunk', 'unverified-doc', 1, 'legacy unverified chunk', 23
                    )
                    """.trimIndent()
                )
            }
        }

        val migrated = Room.databaseBuilder(
            context,
            ArgusDatabase::class.java,
            databaseName
        )
            .addMigrations(migrationWithRetainedCorpus, ArgusDatabase.MIGRATION_2_3)
            .allowMainThreadQueries()
            .build()

        try {
            val database = migrated.openHelper.writableDatabase

            database.query(
                "SELECT `text` FROM `chat_messages` WHERE `id` = 'legacy-chat'"
            ).use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals("Keep me", cursor.getString(0))
            }
            database.query(
                "SELECT `value` FROM `world_memories` WHERE `key` = 'legacy-memory'"
            ).use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals("Keep this memory", cursor.getString(0))
            }
            database.query(
                "SELECT `content`, `documentId` FROM `training_items` " +
                    "WHERE `id` = 'legacy-training'"
            ).use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals("Keep this queued text", cursor.getString(0))
                assertNull(cursor.getString(1))
            }

            assertEquals(3, database.version)

            database.query(
                "SELECT `title`, `rightsStatus` FROM `corpus_documents` WHERE `id` = 'legacy-doc'"
            ).use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals("Legacy article", cursor.getString(0))
                assertEquals("USER_DECLARED", cursor.getString(1))
            }
            database.query(
                "SELECT `rightsStatus`, `userDeclarationText`, " +
                    "`userDeclaredAtEpochMs`, `licenseIdentifier` " +
                    "FROM `corpus_sources` WHERE `id` = 'legacy-source'"
            ).use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals("USER_DECLARED", cursor.getString(0))
                assertEquals(
                    "Legacy explicit user approval recorded at import time.",
                    cursor.getString(1)
                )
                assertEquals(300L, cursor.getLong(2))
                assertNull(cursor.getString(3))
            }
            database.query(
                "SELECT `content` FROM `corpus_chunks` WHERE `id` = 'legacy-chunk'"
            ).use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals("legacy chunk content", cursor.getString(0))
            }

            database.query(
                "SELECT `rightsStatus` FROM `corpus_documents` WHERE `id` = 'unverified-doc'"
            ).use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals("UNVERIFIED", cursor.getString(0))
            }
            database.query(
                "SELECT `rightsStatus` FROM `corpus_sources` WHERE `id` = 'unverified-source'"
            ).use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals("UNVERIFIED", cursor.getString(0))
            }
            database.query(
                "SELECT `content` FROM `corpus_chunks` WHERE `id` = 'unverified-chunk'"
            ).use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals("legacy unverified chunk", cursor.getString(0))
            }
            database.query("SELECT COUNT(*) FROM `corpus_documents`").use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals(2, cursor.getInt(0))
            }
            database.query("SELECT COUNT(*) FROM `corpus_sources`").use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals(2, cursor.getInt(0))
            }
            database.query("SELECT COUNT(*) FROM `corpus_chunks`").use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals(2, cursor.getInt(0))
            }

            val sourceColumns = buildSet {
                database.query("PRAGMA table_info(`corpus_sources`)").use { cursor ->
                    val nameIndex = cursor.getColumnIndexOrThrow("name")
                    while (cursor.moveToNext()) {
                        add(cursor.getString(nameIndex))
                    }
                }
            }
            assertTrue(sourceColumns.containsAll(
                setOf(
                    "licenseIdentifier",
                    "licenseName",
                    "licenseUrl",
                    "attributionText",
                    "rightsEvidenceUrl",
                    "userDeclarationText",
                    "userDeclaredAtEpochMs",
                    "reviewedAtEpochMs"
                )
            ))
        } finally {
            migrated.close()
            context.deleteDatabase(databaseName)
        }
    }
}
