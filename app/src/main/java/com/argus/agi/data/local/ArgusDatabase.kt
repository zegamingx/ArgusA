package com.argus.agi.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.argus.agi.data.local.dao.ChatMessageDao
import com.argus.agi.data.local.dao.TrainingQueueDao
import com.argus.agi.data.local.dao.WorldMemoryDao
import com.argus.agi.data.local.entity.ChatMessageEntity
import com.argus.agi.data.local.entity.TrainingItemEntity
import com.argus.agi.data.local.entity.WorldMemoryEntity

@Database(
    entities = [
        ChatMessageEntity::class,
        WorldMemoryEntity::class,
        TrainingItemEntity::class
    ],
    version = 1,
    exportSchema = false
)
abstract class ArgusDatabase : RoomDatabase() {
    abstract fun chatMessageDao(): ChatMessageDao
    abstract fun worldMemoryDao(): WorldMemoryDao
    abstract fun trainingQueueDao(): TrainingQueueDao

    companion object {
        @Volatile
        private var INSTANCE: ArgusDatabase? = null

        fun getInstance(context: Context): ArgusDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    ArgusDatabase::class.java,
                    "argus_ondevice.db"
                ).fallbackToDestructiveMigration()
                 .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
