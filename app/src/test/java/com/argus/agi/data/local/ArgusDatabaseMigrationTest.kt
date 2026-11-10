package com.argus.agi.data.local

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ArgusDatabaseMigrationTest {

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

        val migrated = Room.databaseBuilder(
            context,
            ArgusDatabase::class.java,
            databaseName
        )
            .addMigrations(ArgusDatabase.MIGRATION_1_2)
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

            assertEquals(2, database.version)
            database.query("SELECT COUNT(*) FROM `corpus_documents`").use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals(0, cursor.getInt(0))
            }
            database.query("SELECT COUNT(*) FROM `corpus_sources`").use { cursor ->
                assertTrue(cursor.moveToFirst())
                assertEquals(0, cursor.getInt(0))
            }
        } finally {
            migrated.close()
            context.deleteDatabase(databaseName)
        }
    }
}
